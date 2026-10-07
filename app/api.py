"""HTTP-сервис beauty-ассистента на FastAPI.

Эндпоинты:
    POST /chat              — реплика в диалоге; session_id связывает реплики одного пользователя
    POST /search            — поиск по каталогу без LLM (удобно для отладки ретривала)
    GET  /products/{id}     — карточка товара
    GET  /health            — состояние сервиса: модель, каталог, доступность Ollama

Запуск из корня проекта:
    uvicorn app.api:app --reload
Документация: http://127.0.0.1:8000/docs

Почему агент вызывается через ainvoke: пока LLM думает (5–40 секунд на ход),
сервер не блокируется и обслуживает другие запросы. Синхронные части (поиск в Qdrant,
эмбеддинги) LangGraph сам уводит в пул потоков.
"""
import json
import os
import time
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from langchain_core.messages import HumanMessage, ToolMessage
from pydantic import BaseModel, Field

from app.agent import MODEL, RECURSION_LIMIT, build_graph
from app.ingest import COLLECTION, EMBED_MODEL, get_client, get_embedder
from app.search import search_products
from app.tools import ACTIVE_SYNONYMS, ACTIVES, CATEGORIES, CATEGORY_SYNONYMS, SKIN_TYPES, _catalog_by_id, normalize

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")


# ---------- схемы запросов и ответов ----------

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, examples=["что-нибудь от прыщей, кожа жирная, до 2000"])
    session_id: str | None = Field(None, description="Пусто — начать новый диалог")


class ToolCall(BaseModel):
    name: str
    args: dict


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    tool_calls: list[ToolCall] = Field(description="Какие инструменты вызвал агент в этом ходе — для отладки")
    product_ids: list[str] = Field(description="Товары, которые вернул поиск в этом ходе")
    verify_retries: int = Field(description="Сколько раз guardrail вернул ответ на переписывание")
    latency_ms: int


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, examples=["что-то от покраснений"])
    skin_type: str | None = Field(None, examples=["чувствительная"])
    category: str | None = None
    active: str | None = None
    max_price: int | None = Field(None, gt=0)
    fragrance_free: bool | None = None
    limit: int = Field(5, ge=1, le=20)


class Product(BaseModel):
    id: str
    name: str
    category: str
    price_rub: int
    actives: list[str]
    concerns: list[str]
    skin_types: list[str]
    fragrance_free: bool
    flags: list[str]
    score: float | None = None


class SearchResponse(BaseModel):
    filters_used: dict
    products: list[Product]


# ---------- приложение ----------

def create_app(llm=None) -> FastAPI:
    """llm можно подменить — так API тестируется без Ollama."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Тяжёлое — один раз при старте, а не на первом запросе пользователя.
        app.state.agent = build_graph(llm)
        await run_in_threadpool(get_embedder)   # модель эмбеддингов ~2 ГБ грузится несколько секунд
        await run_in_threadpool(get_client)
        yield

    app = FastAPI(
        title="Beauty Assistant API",
        description="LLM-ассистент по подбору уходовой косметики: LangGraph-агент, поиск в Qdrant, guardrails.",
        version="0.4.0",
        lifespan=lifespan,
    )

    @app.post("/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest) -> ChatResponse:
        session_id = req.session_id or uuid.uuid4().hex
        config = {"configurable": {"thread_id": session_id}, "recursion_limit": RECURSION_LIMIT}
        t0 = time.perf_counter()
        try:
            result = await app.state.agent.ainvoke({"messages": [HumanMessage(req.message)]}, config)
        except (httpx.ConnectError, ConnectionError):
            raise HTTPException(503, "LLM недоступна: проверьте, что Ollama запущена")

        messages = result["messages"]
        last_user = max(i for i, m in enumerate(messages) if m.type == "human" and m.name != "verifier")
        turn = messages[last_user + 1:]

        product_ids = []
        for m in turn:
            if isinstance(m, ToolMessage) and m.name == "search_catalog":
                try:
                    product_ids += [p["id"] for p in json.loads(m.content).get("products", [])]
                except (ValueError, AttributeError):
                    pass

        return ChatResponse(
            session_id=session_id,
            answer=messages[-1].content,
            tool_calls=[ToolCall(name=c["name"], args=c["args"])
                        for m in turn for c in (getattr(m, "tool_calls", None) or [])],
            product_ids=product_ids,
            verify_retries=sum(1 for m in turn if m.name == "verifier"),
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )

    @app.post("/search", response_model=SearchResponse)
    async def search(req: SearchRequest) -> SearchResponse:
        # Те же правила нормализации, что у инструмента агента: «крем» → «крем для лица».
        filters = {
            "skin_type": normalize(req.skin_type, SKIN_TYPES),
            "category": normalize(req.category, CATEGORIES, CATEGORY_SYNONYMS),
            "active": normalize(req.active, ACTIVES, ACTIVE_SYNONYMS),
            "max_price": req.max_price,
            "fragrance_free": req.fragrance_free,
        }
        for key in ("skin_type", "category", "active"):
            if getattr(req, key) and not filters[key]:
                raise HTTPException(422, f"Неизвестное значение {key}: «{getattr(req, key)}»")
        results = await run_in_threadpool(search_products, req.query, limit=req.limit, **filters)
        return SearchResponse(filters_used=filters, products=[Product(**r) for r in results])

    @app.get("/products/{product_id}", response_model=Product)
    async def get_product(product_id: str) -> Product:
        product = _catalog_by_id().get(product_id)
        if product is None:
            raise HTTPException(404, f"Товара {product_id} нет в каталоге")
        return Product(**product)

    @app.get("/health")
    async def health() -> dict:
        try:
            async with httpx.AsyncClient(timeout=2) as client:
                ollama_ok = (await client.get(f"{OLLAMA_URL}/api/tags")).status_code == 200
        except httpx.HTTPError:
            ollama_ok = False
        indexed = await run_in_threadpool(lambda: get_client().count(COLLECTION).count)
        return {
            "status": "ok" if ollama_ok and indexed else "degraded",
            "llm": MODEL,
            "ollama": ollama_ok,
            "embed_model": EMBED_MODEL,
            "products_indexed": indexed,
        }

    return app


app = create_app()
