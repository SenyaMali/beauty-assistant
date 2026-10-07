"""Инструменты, которые LLM вызывает через tool calling.

Главная идея: модель не ищет товары сама и не помнит каталог.
Она только решает, КАКОЙ инструмент вызвать и С КАКИМИ аргументами,
а данные приходят из Qdrant. Так ассистент не выдумывает товары.

Допустимые значения фильтров перечислены в описании инструмента, но модель
всё равно может прислать своё («крем» вместо «крем для лица»). Поэтому аргументы
нормализуются в коде: синоним приводится к значению каталога, неизвестное
значение отбрасывается, и модель получает об этом пометку, а не ошибку.
"""
import json
import re
from functools import lru_cache
from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from app.ingest import CATALOG
from app.search import search_products

SKIN_TYPES = ["сухая", "жирная", "комбинированная", "нормальная", "чувствительная"]
CATEGORIES = [
    "очищающий гель", "пенка для умывания", "тоник", "сыворотка",
    "крем для лица", "крем для век", "солнцезащитный крем", "маска",
]
CATEGORY_SYNONYMS = {
    "крем": "крем для лица", "увлажняющий крем": "крем для лица",
    "гель": "очищающий гель", "гель для умывания": "очищающий гель", "умывалка": "очищающий гель",
    "пенка": "пенка для умывания", "сыворотку": "сыворотка", "серум": "сыворотка",
    "spf": "солнцезащитный крем", "санскрин": "солнцезащитный крем", "солнцезащита": "солнцезащитный крем",
}
ACTIVES = [
    "ниацинамид", "гиалуроновая кислота", "ретинол", "салициловая кислота",
    "гликолевая кислота", "азелаиновая кислота", "витамин C", "церамиды",
    "центелла азиатская", "пантенол", "пептиды", "цинк",
]
ACTIVE_SYNONYMS = {
    "гиалуронка": "гиалуроновая кислота",
    "салицилка": "салициловая кислота", "bha": "салициловая кислота", "aha": "гликолевая кислота",
    "витамин с": "витамин C", "витамин c": "витамин C", "vitamin c": "витамин C",
    "retinol": "ретинол", "niacinamide": "ниацинамид", "центелла": "центелла азиатская", "cica": "центелла азиатская",
}


def normalize(value: str | None, allowed: list[str], synonyms: dict[str, str] | None = None) -> str | None:
    """Приводит значение от модели к значению каталога. None — если сопоставить не удалось."""
    if not value:
        return None
    v = value.strip().lower()
    for a in allowed:
        if a.lower() == v:
            return a
    if synonyms and v in synonyms:
        return synonyms[v]
    # «жирноватая» -> «жирная», «ретинолом» -> «ретинол»: совпадение по первым 4 буквам
    matches = [a for a in allowed if a[:4].lower() == v[:4]]
    return matches[0] if len(matches) == 1 else None

# Поля, которые отдаём модели. Чем меньше лишнего в контексте, тем дешевле и точнее ответ.
VISIBLE_FIELDS = ["id", "name", "category", "price_rub", "volume", "actives",
                  "concerns", "skin_types", "fragrance_free", "flags", "rating"]


def _compact(product: dict) -> dict:
    return {k: product[k] for k in VISIBLE_FIELDS if k in product}


# Основы слов для поиска актива в реплике пользователя: «ретинолом», «с ниацинамидом», «салицилка».
ACTIVE_STEMS = {
    "ретинол": ["ретинол", "retinol"],
    "ниацинамид": ["ниацинамид", "niacinamide"],
    "гиалуроновая кислота": ["гиалурон", "hyaluron"],
    "салициловая кислота": ["салицил", "bha"],
    "гликолевая кислота": ["гликолев", "aha"],
    "азелаиновая кислота": ["азелаин"],
    "витамин C": ["витамин c", "витамин с", "vitamin c"],
    "церамиды": ["церамид", "ceramide"],
    "центелла азиатская": ["центелл", "cica"],
    "пантенол": ["пантенол"],
    "пептиды": ["пептид"],
    "цинк": ["цинк"],
}


def detect_actives(text: str) -> list[str]:
    """Активы, которые пользователь назвал явно. Детерминированно, без LLM."""
    t = text.lower()
    return [a for a, stems in ACTIVE_STEMS.items()
            if any(re.search(rf"\b{re.escape(st)}", t) for st in stems)]


CATEGORY_STEMS = ["пенк", "гел", "тоник", "сыворот", "серум", "крем", "spf", "санскрин", "солнцезащит", "маск"]


def _current_topic_text(messages: list, max_turns: int = 3) -> str:
    """Реплики пользователя, относящиеся к текущей теме.

    Идём от последней реплики к старым и останавливаемся на первой, где назван
    тип средства: с неё началась текущая тема. «А теперь пенку» обрывает тему
    «сыворотка с ретинолом», а «жирная» и «до 3000» её продолжают.
    """
    texts = [m.content for m in messages if m.type == "human" and getattr(m, "name", None) != "verifier"]
    topic = []
    for text in reversed(texts[-max_turns:]):
        topic.append(text)
        if any(stem in text.lower() for stem in CATEGORY_STEMS):
            break
    return " ".join(reversed(topic))


# Тип средства по словам пользователя. Порядок важен: «крем для век» и «солнцезащитный крем»
# проверяются раньше общего «крем».
CATEGORY_PATTERNS = [
    ("крем для век", r"для век|вокруг глаз|под глаза"),
    ("солнцезащитный крем", r"\bspf|санскрин|солнцезащит|от солнца"),
    ("пенка для умывания", r"пенк"),
    ("очищающий гель", r"\bгел[ья]|умывалк"),
    ("тоник", r"тоник"),
    ("сыворотка", r"сыворот|серум"),
    ("маска", r"маск"),
    ("крем для лица", r"\bкрем"),
]

SKIN_PATTERNS = [
    ("сухая", r"\bсух"),
    ("жирная", r"\bжирн(?:ая|ой|ую)|\bжирная"),
    ("комбинированная", r"комбинир"),
    ("нормальная", r"\bнормальн"),
    ("чувствительная", r"чувствительн(?:ая|ой|ую)\b"),
]


def detect_category(text: str) -> str | None:
    t = text.lower()
    for category, pattern in CATEGORY_PATTERNS:
        if re.search(pattern, t):
            return category
    return None


def detect_skin_type(text: str) -> str | None:
    """Тип кожи, если назван ровно один. «Жирный блеск» и «чувствительность» — это проблемы, не тип."""
    t = text.lower()
    found = [skin for skin, pattern in SKIN_PATTERNS if re.search(pattern, t)]
    return found[0] if len(found) == 1 else None


def _recent_user_text(messages: list, n: int = 3) -> str:
    """Последние реплики пользователя: запасной запрос, если модель не передала query."""
    texts = [m.content for m in messages if m.type == "human" and getattr(m, "name", None) != "verifier"]
    return " ".join(texts[-n:])


@tool
def search_catalog(
    query: str = "",
    skin_type: str | None = None,
    category: str | None = None,
    active: str | None = None,
    max_price: int | None = None,
    fragrance_free: bool | None = None,
    messages: Annotated[list, InjectedState("messages")] = None,
) -> str:
    """Ищет товары в каталоге уходовой косметики. Возвращает только товары в наличии.

    Args:
        query: что нужно пользователю своими словами, с терминами каталога.
            Например: "акне и расширенные поры", "обезвоживание, стянутость".
            Разговорные слова переводи в термины: прыщи -> акне, шелушится -> обезвоживание.
        skin_type: тип кожи, если пользователь его назвал. Одно из: сухая, жирная,
            комбинированная, нормальная, чувствительная.
        category: тип средства, только если пользователь назвал его явно. Одно из:
            очищающий гель, пенка для умывания, тоник, сыворотка, крем для лица,
            крем для век, солнцезащитный крем, маска.
        active: активный компонент, если пользователь назвал его явно («с ретинолом»). Одно из:
            ниацинамид, гиалуроновая кислота, ретинол, салициловая кислота, гликолевая кислота,
            азелаиновая кислота, витамин C, церамиды, центелла азиатская, пантенол, пептиды, цинк.
        max_price: верхняя граница цены в рублях, если пользователь назвал бюджет.
        fragrance_free: true, если пользователь просит без отдушек.
    """
    notes = []
    if not query.strip():
        # Модель иногда «забывает» query. Не падаем, а берём смысл из реплик пользователя.
        query = _recent_user_text(messages or [])
        notes.append("query был пустым, искала по последним репликам пользователя")
    act = normalize(active, ACTIVES, ACTIVE_SYNONYMS)
    if active and not act:
        notes.append(f"актив «{active}» не распознан, искала без него")
    if not act:
        # Модель может «забыть» актив. Если пользователь назвал его в недавних репликах — фильтруем сами.
        mentioned = detect_actives(_current_topic_text(messages or []))
        if len(mentioned) == 1:
            act = mentioned[0]
            notes.append(f"пользователь просил «{act}» — фильтр по активу включён автоматически")
    skin = normalize(skin_type, SKIN_TYPES)
    if skin_type and not skin:
        notes.append(f"тип кожи «{skin_type}» не распознан, искала без него")
    if not skin:
        # Тип кожи — свойство человека, а не темы: держится весь недавний диалог.
        skin = detect_skin_type(_recent_user_text(messages or [], n=5))
        if skin:
            notes.append(f"тип кожи «{skin}» взят из реплик пользователя")

    cat = normalize(category, CATEGORIES, CATEGORY_SYNONYMS)
    if category and not cat:
        notes.append(f"категория «{category}» не распознана, искала без неё")
    if messages:
        named_category = detect_category(_current_topic_text(messages))
        if named_category and named_category != cat:
            # Пользователь назвал тип средства — он главнее догадки модели.
            if cat:
                notes.append(f"категория «{cat}» заменена на «{named_category}», которую назвал пользователь")
            else:
                notes.append(f"категория «{named_category}» взята из реплик пользователя")
            cat = named_category
        elif cat and not named_category:
            # Модель сама придумала тип средства, а пользователь его не называл: это сужает поиск зря.
            notes.append(f"категория «{cat}» убрана: пользователь не называл тип средства")
            cat = None

    results = search_products(query, skin_type=skin, category=cat,
                              max_price=max_price, fragrance_free=fragrance_free, active=act, limit=6)
    payload = {"filters_used": {"skin_type": skin, "category": cat, "active": act, "max_price": max_price,
                                "fragrance_free": fragrance_free},
               "products": [_compact(r) for r in results]}
    if not results:
        # Всегда JSON: и модель, и eval видят, с какими фильтрами шёл поиск.
        payload["message"] = ("НИЧЕГО НЕ НАЙДЕНО. Не называй никаких товаров. Скажи пользователю честно "
                              "и предложи ослабить условие: поднять бюджет или убрать тип средства.")
    if notes:
        payload["notes"] = notes
    return json.dumps(payload, ensure_ascii=False)


@lru_cache(maxsize=1)
def _catalog_by_id() -> dict[str, dict]:
    return {p["id"]: p for p in json.loads(CATALOG.read_text(encoding="utf-8"))}


@tool
def get_product(product_id: str) -> str:
    """Возвращает полную карточку товара по id (например, P0042): описание, наличие, все флаги.

    Используй, когда пользователь спрашивает подробности о конкретном товаре из выдачи.
    """
    product = _catalog_by_id().get(product_id)
    if product is None:
        return f"Товара с id {product_id} нет в каталоге."
    return json.dumps(product, ensure_ascii=False)


TOOLS = [search_catalog, get_product]
