"""Тесты API и guardrails без Ollama и без скачивания модели эмбеддингов.

LLM подменяется сценарием ответов, эмбеддинги — детерминированными векторами из хеша текста.
Так проверяется механика сервиса (маршрутизация графа, фильтры, verify, triage, ошибки),
а не качество модели — качество меряет eval/.

Запуск: python -m pytest -q
"""
import hashlib
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

import app.ingest as ingest
import app.search as search


def fake_embed(texts, **_):
    vectors = []
    for t in texts:
        v = np.random.default_rng(int(hashlib.md5(t.encode()).hexdigest()[:8], 16)).random(16)
        vectors.append((v / np.linalg.norm(v)).tolist())
    return vectors


class ScriptedLLM(GenericFakeChatModel):
    """Отвечает заранее заданными сообщениями; bind_tools ничего не меняет."""

    def bind_tools(self, tools, **kwargs):
        return self


@pytest.fixture(scope="session", autouse=True)
def index(tmp_path_factory):
    ingest.embed = fake_embed
    search.embed = fake_embed
    ingest.get_embedder = lambda: None
    ingest.DB_PATH = tmp_path_factory.mktemp("qdrant")
    ingest._client = None
    ingest.main()


def make_client(script):
    import app.api as api
    api.get_embedder = lambda: None
    return TestClient(api.create_app(ScriptedLLM(messages=iter(script))))


def search_call(**args):
    return AIMessage("", tool_calls=[{"name": "search_catalog", "args": args, "id": "call-1"}])


def first_product(**args):
    from app.tools import search_catalog
    return json.loads(search_catalog.invoke({**args, "messages": []}))["products"][0]


def test_dialog_keeps_session_and_calls_search():
    args = {"query": "акне", "skin_type": "жирная", "max_price": 2000}
    p = first_product(**args)
    with make_client([AIMessage("Какой у вас тип кожи?"), search_call(**args),
                      AIMessage(f"**{p['name']}** — {p['price_rub']} ₽")]) as client:
        first = client.post("/chat", json={"message": "что-нибудь от прыщей"}).json()
        assert first["tool_calls"] == []
        second = client.post("/chat", json={"message": "жирная, до 2000",
                                            "session_id": first["session_id"]}).json()
    assert [c["name"] for c in second["tool_calls"]] == ["search_catalog"]
    assert p["id"] in second["product_ids"]
    assert second["verify_retries"] == 0


def test_invented_product_is_rejected_by_verify():
    args = {"query": "акне", "skin_type": "жирная", "max_price": 2000}
    p = first_product(**args)
    with make_client([search_call(**args), AIMessage("Крем Neova — 1500 ₽"),
                      AIMessage(f"**{p['name']}** — {p['price_rub']} ₽")]) as client:
        r = client.post("/chat", json={"message": "жирная кожа, прыщи, до 2000"}).json()
    assert r["verify_retries"] == 1
    assert "Neova" not in r["answer"]


def test_medical_red_flags_skip_llm():
    with make_client([AIMessage("LLM не должна вызываться")]) as client:
        r = client.post("/chat", json={"message": "пятно на щеке быстро растёт и кровит"}).json()
    assert "дерматолог" in r["answer"]
    assert r["tool_calls"] == []


def test_search_normalizes_and_filters():
    with make_client([]) as client:
        r = client.post("/search", json={"query": "покраснения", "skin_type": "чувствительная",
                                         "category": "крем", "max_price": 3000}).json()
    assert r["filters_used"]["category"] == "крем для лица"
    assert all(p["category"] == "крем для лица" and p["price_rub"] <= 3000
               and "чувствительная" in p["skin_types"] for p in r["products"])


def test_search_rejects_unknown_filter_value():
    with make_client([]) as client:
        r = client.post("/search", json={"query": "x", "skin_type": "зелёная"})
    assert r.status_code == 422


def test_product_lookup():
    with make_client([]) as client:
        assert client.get("/products/P0213").status_code == 200
        assert client.get("/products/NOPE").status_code == 404


def test_ollama_down_returns_503(monkeypatch):
    monkeypatch.setenv("OLLAMA_URL", "http://127.0.0.1:9")
    import app.api as api
    api.get_embedder = lambda: None
    with TestClient(api.create_app()) as client:
        r = client.post("/chat", json={"message": "привет"})
    assert r.status_code == 503
