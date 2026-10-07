"""Генерирует синтетический каталог уходовой косметики.

Бренды выдуманные, цены и составы — правдоподобные, но не реальные.
Запуск: python data/generate_catalog.py  ->  data/catalog.json
Генерация детерминированная (seed), чтобы каталог и eval-набор не расходились.
"""
import json
import random
from pathlib import Path

random.seed(42)

BRANDS = ["Lumea", "Derma Nord", "Kora Lab", "Sela", "Mirai Skin",
          "Botanica Pura", "Clearis", "Avella", "Oslo Derm", "Hanami"]

# категория -> (диапазон цен в рублях, объёмы)
CATEGORIES = {
    "очищающий гель": ((590, 1890), ["150 мл", "200 мл"]),
    "пенка для умывания": ((490, 1590), ["150 мл"]),
    "тоник": ((690, 2190), ["150 мл", "200 мл"]),
    "сыворотка": ((1290, 4990), ["30 мл"]),
    "крем для лица": ((1190, 4590), ["50 мл"]),
    "крем для век": ((1390, 3990), ["15 мл"]),
    "солнцезащитный крем": ((890, 2990), ["50 мл"]),
    "маска": ((790, 2490), ["75 мл", "100 мл"]),
}

# актив -> (решаемые проблемы, подходящие типы кожи, флаги)
ACTIVES = {
    "ниацинамид": (["расширенные поры", "акне", "неровный тон"],
                   ["жирная", "комбинированная", "нормальная"], set()),
    "гиалуроновая кислота": (["обезвоживание"],
                             ["сухая", "нормальная", "комбинированная", "чувствительная"], set()),
    "ретинол": (["морщины", "неровный тон", "акне"],
                ["нормальная", "жирная", "комбинированная"], {"не при беременности", "вечернее применение"}),
    "салициловая кислота": (["акне", "расширенные поры"],
                            ["жирная", "комбинированная"], {"кислота", "не при беременности"}),
    "гликолевая кислота": (["неровный тон", "пигментация"],
                           ["нормальная", "жирная", "комбинированная"], {"кислота"}),
    "азелаиновая кислота": (["покраснения", "акне", "пигментация"],
                            ["чувствительная", "жирная", "комбинированная"], set()),
    "витамин C": (["пигментация", "неровный тон"],
                  ["нормальная", "сухая", "комбинированная"], {"утреннее применение"}),
    "церамиды": (["обезвоживание", "чувствительность"],
                 ["сухая", "чувствительная", "нормальная"], set()),
    "центелла азиатская": (["покраснения", "чувствительность"],
                           ["чувствительная", "сухая", "комбинированная"], set()),
    "пантенол": (["покраснения", "обезвоживание"],
                 ["чувствительная", "сухая", "нормальная"], set()),
    "пептиды": (["морщины", "потеря упругости"],
                ["нормальная", "сухая"], set()),
    "цинк": (["акне", "жирный блеск"],
             ["жирная", "комбинированная"], set()),
}

# какие активы уместны в какой категории
CATEGORY_ACTIVES = {
    "очищающий гель": ["салициловая кислота", "цинк", "центелла азиатская", "пантенол"],
    "пенка для умывания": ["церамиды", "центелла азиатская", "цинк"],
    "тоник": ["гликолевая кислота", "салициловая кислота", "ниацинамид", "пантенол"],
    "сыворотка": list(ACTIVES),
    "крем для лица": ["церамиды", "пептиды", "ниацинамид", "гиалуроновая кислота", "ретинол", "центелла азиатская"],
    "крем для век": ["пептиды", "гиалуроновая кислота", "ретинол"],
    "солнцезащитный крем": ["ниацинамид", "центелла азиатская", "гиалуроновая кислота"],
    "маска": ["цинк", "центелла азиатская", "гиалуроновая кислота", "гликолевая кислота"],
}

NAME_WORDS = ["Balance", "Calm", "Glow", "Pure", "Hydra", "Renew", "Clear", "Soft", "Bright", "Barrier"]


def round_price(p: int) -> int:
    return int(round(p / 100) * 100 - 10)


def make_product(idx: int) -> dict:
    category = random.choice(list(CATEGORIES))
    (lo, hi), volumes = CATEGORIES[category]
    n_actives = 1 if category in ("пенка для умывания", "солнцезащитный крем") else random.choice([1, 2, 2, 3])
    actives = random.sample(CATEGORY_ACTIVES[category], k=min(n_actives, len(CATEGORY_ACTIVES[category])))

    concerns, skin_types, flags = set(), None, set()
    for a in actives:
        c, s, f = ACTIVES[a]
        concerns.update(c)
        skin_types = set(s) if skin_types is None else skin_types & set(s)
        flags |= f
    if not skin_types:  # активы не сошлись по типу кожи — берём самый безопасный вариант
        skin_types = {"нормальная"}

    fragrance_free = random.random() < 0.55
    if "чувствительная" in skin_types and not fragrance_free:
        skin_types.discard("чувствительная") if len(skin_types) > 1 else None
    if category == "солнцезащитный крем":
        flags.add(random.choice(["SPF 30", "SPF 50"]))

    brand = random.choice(BRANDS)
    name = f"{brand} {random.choice(NAME_WORDS)} {category}"
    price = round_price(random.randint(lo, hi))

    description = (
        f"{category.capitalize()} с активами: {', '.join(actives)}. "
        f"Помогает при проблемах: {', '.join(sorted(concerns))}. "
        f"Подходит для кожи: {', '.join(sorted(skin_types))}."
        + (" Без отдушек." if fragrance_free else "")
        + (" " + "; ".join(sorted(flags)).capitalize() + "." if flags else "")
    )

    return {
        "id": f"P{idx:04d}",
        "name": name,
        "brand": brand,
        "category": category,
        "price_rub": price,
        "volume": random.choice(volumes),
        "actives": actives,
        "concerns": sorted(concerns),
        "skin_types": sorted(skin_types),
        "fragrance_free": fragrance_free,
        "flags": sorted(flags),
        "rating": round(random.uniform(3.6, 4.9), 1),
        "in_stock": random.random() < 0.85,
        "description": description,
    }


def main(n: int = 250) -> None:
    products = [make_product(i + 1) for i in range(n)]
    out = Path(__file__).parent / "catalog.json"
    out.write_text(json.dumps(products, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Записано {len(products)} товаров в {out}")


if __name__ == "__main__":
    main()
