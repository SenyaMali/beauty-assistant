"""Поиск по каталогу: семантика + жёсткие фильтры.

Именно эта функция в этапе 2 станет инструментом (tool) для LLM-агента:
модель будет сама заполнять аргументы из диалога с пользователем.

Проверка из корня проекта:
    python -m app.search "что-то от покраснений без отдушек" --skin чувствительная --max-price 2500
"""
import argparse

from qdrant_client import models

from app.ingest import COLLECTION, embed, get_client


def search_products(
    query: str,
    skin_type: str | None = None,
    category: str | None = None,
    max_price: int | None = None,
    fragrance_free: bool | None = None,
    active: str | None = None,
    limit: int = 5,
) -> list[dict]:
    must = [models.FieldCondition(key="in_stock", match=models.MatchValue(value=True))]
    if skin_type:
        must.append(models.FieldCondition(key="skin_types", match=models.MatchValue(value=skin_type)))
    if category:
        must.append(models.FieldCondition(key="category", match=models.MatchValue(value=category)))
    if max_price:
        must.append(models.FieldCondition(key="price_rub", range=models.Range(lte=max_price)))
    if fragrance_free is not None:
        must.append(models.FieldCondition(key="fragrance_free", match=models.MatchValue(value=fragrance_free)))
    if active:
        must.append(models.FieldCondition(key="actives", match=models.MatchValue(value=active)))

    hits = get_client().query_points(
        COLLECTION,
        query=embed([query], kind="query")[0],
        query_filter=models.Filter(must=must),
        limit=limit,
    ).points
    return [{**h.payload, "score": round(h.score, 3)} for h in hits]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--skin")
    parser.add_argument("--category")
    parser.add_argument("--max-price", type=int)
    parser.add_argument("--no-fragrance", action="store_true")
    parser.add_argument("--active")
    args = parser.parse_args()

    results = search_products(
        args.query,
        skin_type=args.skin,
        category=args.category,
        max_price=args.max_price,
        fragrance_free=True if args.no_fragrance else None,
        active=args.active,
    )
    if not results:
        print("Ничего не найдено — попробуйте ослабить фильтры.")
    for r in results:
        print(f"{r['score']:.3f}  {r['price_rub']:>5} ₽  {r['name']}  [{', '.join(r['actives'])}]")


if __name__ == "__main__":
    main()
