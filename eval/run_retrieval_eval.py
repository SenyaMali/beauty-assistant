"""Оценка поиска без LLM: находит ли векторный поиск нужные товары.

Разметка не ручная: каталог структурированный, поэтому «релевантный товар»
задаётся правилом (например, «в concerns есть акне»). Так эталон точный и
воспроизводимый, а кейсы легко добавлять.

Метрики на кейс (по топ-5 выдачи):
  precision@5 — доля релевантных товаров в топ-5;
  MRR        — 1 / позиция первого релевантного (1.0 = релевантный первым, 0 = ни одного);
  baseline   — доля релевантных среди всех товаров, прошедших фильтры: столько P@5
               в среднем дал бы случайный поиск. Модель полезна настолько, насколько её
               P@5 выше baseline. Без этой поправки «лёгкие» кейсы (где релевантна
               половина каталога) завышают общий результат.

Индекс строится в памяти отдельно для каждой модели, рабочая база qdrant_data не трогается.

Запуск из корня проекта:
    python -m eval.run_retrieval_eval                      # модель из app/ingest.py
    python -m eval.run_retrieval_eval --models all         # все три модели (e5-large ~2.2 ГБ)
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

from app.ingest import CATALOG, EMBED_MODEL, load_embedder, prefixes

ROOT = Path(__file__).resolve().parent.parent
CASES = Path(__file__).parent / "retrieval_cases.json"
RESULTS = Path(__file__).parent / "results"
K = 5

ALL_MODELS = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "intfloat/multilingual-e5-large",
]


def is_relevant(product: dict, rule: dict) -> bool:
    checks = []
    if "concerns_any" in rule:
        checks.append(bool(set(product["concerns"]) & set(rule["concerns_any"])))
    if "category_any" in rule:
        checks.append(product["category"] in rule["category_any"])
    if "actives_any" in rule:
        checks.append(bool(set(product["actives"]) & set(rule["actives_any"])))
    return all(checks)


def build_filter(filters: dict | None) -> models.Filter:
    must = [models.FieldCondition(key="in_stock", match=models.MatchValue(value=True))]
    for key, value in (filters or {}).items():
        field = "skin_types" if key == "skin_type" else key
        must.append(models.FieldCondition(key=field, match=models.MatchValue(value=value)))
    return models.Filter(must=must)


def matches_filters(product: dict, filters: dict | None) -> bool:
    if not product["in_stock"]:
        return False
    for key, value in (filters or {}).items():
        field = product["skin_types"] if key == "skin_type" else product[key]
        if (value not in field) if isinstance(field, list) else (field != value):
            return False
    return True


def evaluate(model: str, products: list[dict], cases: list[dict]) -> dict:
    q_prefix, p_prefix = prefixes(model)
    embedder = load_embedder(model)

    t0 = time.perf_counter()
    vectors = [v.tolist() for v in embedder.embed([p_prefix + p["description"] for p in products])]
    index_sec = time.perf_counter() - t0

    client = QdrantClient(":memory:")
    client.create_collection("eval", vectors_config=models.VectorParams(
        size=len(vectors[0]), distance=models.Distance.COSINE))
    client.upsert("eval", points=[models.PointStruct(id=i, vector=v, payload=p)
                                  for i, (v, p) in enumerate(zip(vectors, products))])

    rows = []
    for case in cases:
        qv = next(iter(embedder.embed([q_prefix + case["query"]]))).tolist()
        hits = client.query_points("eval", query=qv, query_filter=build_filter(case.get("filters")), limit=K).points
        rel = [is_relevant(h.payload, case["relevant"]) for h in hits]
        first = next((i for i, r in enumerate(rel) if r), None)
        pool = [p for p in products if matches_filters(p, case.get("filters"))]
        rows.append({
            "id": case["id"], "query": case["query"],
            "baseline": sum(is_relevant(p, case["relevant"]) for p in pool) / len(pool),
            "precision@5": sum(rel) / K,
            "mrr": 0.0 if first is None else 1 / (first + 1),
            "top5": [f'{h.payload["category"]} [{", ".join(h.payload["concerns"])}]' for h in hits],
        })
    client.close()

    n = len(rows)
    return {
        "model": model,
        "precision@5": round(sum(r["precision@5"] for r in rows) / n, 3),
        "baseline": round(sum(r["baseline"] for r in rows) / n, 3),
        "lift": round(sum(r["precision@5"] - r["baseline"] for r in rows) / n, 3),
        "mrr": round(sum(r["mrr"] for r in rows) / n, 3),
        "index_sec": round(index_sec, 1),
        "cases": rows,
    }


def print_report(results: list[dict]) -> None:
    short = lambda m: m.split("/")[-1]
    print("\n" + "=" * 72)
    print(f"{'модель':<45}{'P@5':>7}{'случайно':>10}{'прирост':>9}{'MRR':>7}{'индекс, с':>11}")
    for r in results:
        print(f"{short(r['model']):<45}{r['precision@5']:>7.2f}{r['baseline']:>10.2f}"
              f"{r['lift']:>+9.2f}{r['mrr']:>7.2f}{r['index_sec']:>11}")

    print("\nПо кейсам (P@5; в скобках — случайный поиск):")
    header = "".join(f"{short(r['model'])[:14]:>16}" for r in results)
    print(f"{'кейс':<6}{'запрос':<36}{'(случ.)':>8}{header}")
    for i, case in enumerate(results[0]["cases"]):
        scores = "".join(f"{r['cases'][i]['precision@5']:>16.2f}" for r in results)
        print(f"{case['id']:<6}{case['query'][:34]:<36}{'(' + format(case['baseline'], '.2f') + ')':>8}{scores}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=EMBED_MODEL,
                        help="имя модели, несколько через запятую или all")
    args = parser.parse_args()
    model_list = ALL_MODELS if args.models == "all" else args.models.split(",")

    products = json.loads(CATALOG.read_text(encoding="utf-8"))
    cases = json.loads(CASES.read_text(encoding="utf-8"))

    results = []
    for m in model_list:
        print(f"→ {m}")
        try:
            results.append(evaluate(m, products, cases))
        except Exception as e:  # одна упавшая модель не должна стирать результаты остальных
            print(f"  ПРОПУЩЕНА: {type(e).__name__}: {str(e)[:200]}")
    if not results:
        raise SystemExit("Ни одна модель не отработала.")
    print_report(results)

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"retrieval_{datetime.now():%Y%m%d_%H%M}.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nПодробности (топ-5 по каждому кейсу): {out}")


if __name__ == "__main__":
    main()
