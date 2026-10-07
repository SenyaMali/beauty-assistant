"""Индексация каталога в Qdrant.

Qdrant работает в embedded-режиме: база лежит в папке qdrant_data/,
отдельный сервер и Docker на этом этапе не нужны.
Эмбеддинги считает fastembed (ONNX, работает на CPU). Модель по умолчанию — multilingual-e5-large
(~2,2 ГБ, скачивается один раз): в eval поиска она дала P@5 0.89 против 0.59 у MiniLM.
Другую модель можно задать переменной EMBED_MODEL; после смены модели нужно переиндексировать.

Запуск из корня проекта: python -m app.ingest
"""
import atexit
import json
import os
from pathlib import Path

from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "data" / "catalog.json"
DB_PATH = ROOT / "qdrant_data"
COLLECTION = "products"
# Выбрана по результатам eval поиска (eval/run_retrieval_eval.py): e5-large — P@5 0.89, MiniLM — 0.59.
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-large")


_client: QdrantClient | None = None
_embedder: TextEmbedding | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(path=str(DB_PATH))
        atexit.register(_close_client)
    return _client


def _close_client() -> None:
    # Закрываем базу и отпускаем ссылку, пока Python ещё жив.
    # Иначе Qdrant закрывается повторно при выключении интерпретатора и сыплет ImportError.
    global _client
    if _client is not None:
        _client.close()
        _client = None


def prefixes(model: str) -> tuple[str, str]:
    """(префикс запроса, префикс документа). Модели e5 обучены с ними: без префиксов качество падает."""
    return ("query: ", "passage: ") if "e5" in model else ("", "")


def load_embedder(model: str) -> TextEmbedding:
    """Загружает модель; для моделей с внешним файлом весов — через локальную копию.

    У multilingual-e5-large веса лежат отдельным файлом model.onnx_data. Кэш HuggingFace
    хранит его как ссылку в общую папку blobs, а свежий onnxruntime запрещает читать
    внешние данные за пределами папки модели («External data path escapes model directory»).
    Обход: скачать модель в обычную папку models/ без ссылок и указать путь явно.
    """
    try:
        return TextEmbedding(model)
    except Exception as e:  # noqa: BLE001 — различаем по тексту ошибки onnxruntime
        if "External data path" not in str(e):
            raise
    from huggingface_hub import snapshot_download

    source = next(m["sources"]["hf"] for m in TextEmbedding.list_supported_models() if m["model"] == model)
    local_dir = ROOT / "models" / source.replace("/", "__")
    if not local_dir.exists():
        print(f"Веса во внешнем файле — копирую модель в {local_dir.relative_to(ROOT)} (один раз)")
    snapshot_download(source, local_dir=str(local_dir))
    return TextEmbedding(model, specific_model_path=str(local_dir))


def get_embedder() -> TextEmbedding:
    global _embedder
    if _embedder is None:
        _embedder = load_embedder(EMBED_MODEL)
    return _embedder


def embed(texts: list[str], kind: str = "passage") -> list[list[float]]:
    """kind: "query" для запроса пользователя, "passage" для описаний товаров."""
    q_prefix, p_prefix = prefixes(EMBED_MODEL)
    prefix = q_prefix if kind == "query" else p_prefix
    return [v.tolist() for v in get_embedder().embed([prefix + t for t in texts])]


def main() -> None:
    products = json.loads(CATALOG.read_text(encoding="utf-8"))
    client = get_client()

    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)

    vectors = embed([p["description"] for p in products], kind="passage")
    client.create_collection(
        COLLECTION,
        vectors_config=models.VectorParams(size=len(vectors[0]), distance=models.Distance.COSINE),
    )
    # Вектор строится по описанию; структурные поля уходят в payload для фильтров.
    client.upsert(
        COLLECTION,
        points=[models.PointStruct(id=i, vector=v, payload=p) for i, (v, p) in enumerate(zip(vectors, products))],
    )

    # Индексы payload: по ним фильтруем точно, а не надеемся на семантику.
    # В локальном режиме Qdrant их игнорирует (и предупреждает), но на этапе 4
    # мы переедем на Qdrant-сервер в Docker, и там они ускорят фильтрацию.
    import warnings
    warnings.filterwarnings("ignore", message="Payload indexes have no effect")
    for field, schema in [
        ("category", models.PayloadSchemaType.KEYWORD),
        ("skin_types", models.PayloadSchemaType.KEYWORD),
        ("concerns", models.PayloadSchemaType.KEYWORD),
        ("price_rub", models.PayloadSchemaType.INTEGER),
        ("in_stock", models.PayloadSchemaType.BOOL),
        ("fragrance_free", models.PayloadSchemaType.BOOL),
    ]:
        client.create_payload_index(COLLECTION, field_name=field, field_schema=schema)

    print(f"Проиндексировано {client.count(COLLECTION).count} товаров, модель {EMBED_MODEL}")


if __name__ == "__main__":
    main()
