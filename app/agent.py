"""LangGraph-агент beauty-ассистента.

    START -> triage --(тревожные симптомы)--> END (готовый ответ «к дерматологу»)
                 \\--(обычный запрос)--> assistant --(есть tool_calls)--> tools -> assistant -> ...
                      \\--(текстовый ответ)--> verify --(ок)--> END
                                                    \\--(ошибка)--> assistant (ещё попытка)

triage    — детерминированная проверка до LLM: при признаках, похожих на заболевание
            (кровит, гноится, быстро растёт, болит), агент не подбирает косметику вовсе.
            Модель здесь не участвует: в медицинском случае цена ошибки слишком высока,
            а 7B в eval советовала врача и тут же предлагала кремы.
assistant — вызов LLM. Модель либо отвечает текстом, либо просит вызвать инструмент.
tools     — выполняет инструменты и кладёт результат в историю.
verify    — guardrail: проверяет ответ кодом, а не просьбой в промпте.
            1) Все цены в ответе должны быть из результатов инструментов этого хода.
               Цена, которой нет в выдаче, = выдуманный товар.
            2) Каждый актив, названный в ответе, есть в выдаче этого хода
               (иначе модель приписывает товару состав, о котором спросили).
            3) Ответ на русском (Qwen иногда переходит на китайский).
            Если проверка не прошла, модель получает замечание и пробует снова.
            После MAX_RETRIES отдаём безопасный ответ-заглушку.

Чат в терминале: python -m app.agent
Нужен запущенный Ollama с моделью: ollama pull qwen2.5:7b
"""
import json
import os
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from app.tools import TOOLS, detect_actives, detect_category, detect_skin_type

MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
RECURSION_LIMIT = 20  # защита от бесконечного цикла вызовов
MAX_RETRIES = 2

SYSTEM_PROMPT = """Ты — beauty-ассистент магазина уходовой косметики. Помогаешь подобрать средства.

Как работать:
1. Нужно знать тип кожи или что беспокоит. Если неизвестно ни то, ни другое — задай ОДИН короткий уточняющий вопрос и не ищи.
2. Когда данных достаточно, вызови search_catalog. В query ВСЕГДА пиши, что ищет пользователь, с учётом всего диалога. Тип кожи, бюджет, «без отдушек» передавай фильтрами. Категорию и актив (ретинол, ниацинамид…) передавай, только если пользователь их назвал. Помни условия из прошлых реплик: если раньше просили ретинол, он остаётся в фильтре.
3. Предлагай до 3 товаров СТРОГО из результатов инструмента: название и цена — ровно как в результатах. Никогда не придумывай товары, бренды, цены и составы.
4. Для каждого товара: название, цена в ₽, одна строка — почему подходит этому человеку. Состав называй только из поля actives этого товара.
5. Если во флагах есть «не при беременности» или «кислота» — коротко предупреди.
6. Если инструмент ничего не нашёл — скажи честно и предложи ослабить условие. Товары не называй.
7. Про конкретный товар из выдачи вызывай get_product с его id из результатов.

Ты не врач. При признаках кожного заболевания (сильное воспаление, боль, быстро меняющееся пятно) советуй дерматолога.
Отвечай ТОЛЬКО на русском языке, коротко и дружелюбно."""

# Признаки, при которых нужен врач, а не косметика. Ловим основы слов во всех формах.
RED_FLAGS = re.compile(
    r"кровит|кровоточ|кровь|гно[йияеё]|гноит|язв|"
    r"быстро (?:растёт|растет|увеличива)|растёт пятно|растет пятно|"
    r"родинк\w* (?:измени|растёт|растет|темнеет|чешется|кровит)|"
    r"болит|сильная боль|опух|отёк|отек|температур|ожог",
    re.IGNORECASE,
)
MEDICAL_ANSWER = (
    "То, что вы описываете, похоже на повод показаться врачу, а не подбирать уход. "
    "Пожалуйста, обратитесь к дерматологу: при кровоточивости, быстром росте пятна, "
    "боли или гное важна очная диагностика. Косметику до визита к врачу лучше не наносить "
    "на это место. Если захотите подобрать уход для остальной кожи — я помогу."
)

FALLBACK = ("Не получилось подобрать товары надёжно. Уточните, пожалуйста, тип кожи, "
            "что беспокоит и бюджет — попробую ещё раз.")

CJK = re.compile(r"[぀-ヿ一-鿿]")
PRICE = re.compile(r"(\d[\d\s ]{2,6})\s*(?:₽|руб)")


def _current_turn(messages: list) -> list:
    """Сообщения после последней реплики пользователя (замечания проверки — не реплики)."""
    last_human = max(i for i, m in enumerate(messages) if m.type == "human" and m.name != "verifier")
    return messages[last_human + 1:]


def _allowed_prices(messages: list, turn: list) -> set[int]:
    """Цены, которые можно называть: товары из выдачи этого хода и бюджеты.

    Бюджет («до 2000 ₽») — не выдуманный товар: его называл пользователь
    или модель передала в фильтр max_price.
    """
    prices = set()
    for m in messages:
        if m.type == "human" and m.name != "verifier":
            prices |= {int(n) for n in re.findall(r"\d{3,6}", m.content.replace(" ", ""))}
    for m in turn:
        for call in getattr(m, "tool_calls", None) or []:
            if call["args"].get("max_price"):
                prices.add(int(call["args"]["max_price"]))
    # Товары из ЛЮБОГО поиска в диалоге, а не только этого хода: модель вправе сослаться
    # на выдачу прошлого хода («до 3000» → выбрать из уже найденного). Это реальные товары
    # каталога, не выдумка. Проблему нашёл eval: 14B в A04 отвечала верно и получала отказ.
    for m in messages:
        if not isinstance(m, ToolMessage):
            continue
        try:
            data = json.loads(m.content)
        except (json.JSONDecodeError, TypeError):
            continue  # «ничего не найдено» и прочий текст
        products = data.get("products", [data]) if isinstance(data, dict) else data
        prices |= {p["price_rub"] for p in products if isinstance(p, dict) and "price_rub" in p}
    return prices


def _allowed_actives(turn: list) -> set[str] | None:
    """Активы из ответов инструментов этого хода (составы товаров и фильтры).

    None — инструменты в этом ходе не вызывались (например, уточняющий вопрос
    «вам нужен именно ретинол?»): товаров нет, приписывать состав некому.
    """
    tool_outputs = [m.content for m in turn if isinstance(m, ToolMessage)]
    if not tool_outputs:
        return None
    found_products = any(f'"price_rub"' in text for text in tool_outputs)
    if not found_products:
        # Поиск ничего не нашёл — товаров нет, приписывать состав некому. Совет вида
        # «попробуйте средства с салициловой кислотой» без товаров и цен — не выдумка.
        return None
    return {a for text in tool_outputs for a in detect_actives(text)}


def _needs_new_search(messages: list, turn: list) -> bool:
    """Пользователь назвал новое условие, а агент советует товары без нового поиска.

    Старая выдача — максимум 6 товаров по старым условиям; с новым бюджетом или типом
    кожи могут подойти совсем другие. Нашёл eval: 14B в A04 на «до 3000» выбрала из
    прошлой выдачи и назвала сыворотку за 3390 ₽.
    """
    if any(isinstance(m, ToolMessage) for m in turn):
        return False
    answer = messages[-1].content
    if not PRICE.search(answer):  # не советует товары — например, уточняет
        return False
    last_user = next(m.content for m in reversed(messages) if m.type == "human" and m.name != "verifier")
    had_search_before = any(isinstance(m, ToolMessage) and m.name == "search_catalog" for m in messages)
    new_condition = bool(
        re.search(r"\d{3,6}", last_user.replace(" ", ""))
        or detect_skin_type(last_user) or detect_category(last_user) or detect_actives(last_user)
        or re.search(r"без отдуш", last_user.lower())
    )
    return had_search_before and new_condition


def check_answer(answer: str, allowed_prices: set[int], allowed_actives: set[str] | None = None) -> str | None:
    """None — ответ прошёл проверку, иначе текст замечания для модели."""
    if CJK.search(answer):
        return "Ответ должен быть полностью на русском языке. Перепиши ответ на русском."
    if allowed_actives is not None:
        unsupported = set(detect_actives(answer)) - allowed_actives
        if unsupported:
            return (f"В ответе упомянуты {sorted(unsupported)}, но их нет в составе найденных товаров. "
                    "Называй состав строго из поля actives. Если нужного компонента нет в выдаче — "
                    "так и скажи, а не приписывай его товарам.")
    mentioned = {int(re.sub(r"\D", "", p)) for p in PRICE.findall(answer)}
    invented = mentioned - allowed_prices
    if invented:
        return (f"Цен {sorted(invented)} нет в результатах инструмента — это выдуманные товары. "
                "Называй только товары из результатов search_catalog, с их точными названиями и ценами. "
                "Если инструмент ничего не нашёл — честно скажи об этом и не называй товары.")
    return None


def build_graph(llm=None):
    """llm можно подменить — так граф тестируется без Ollama."""
    llm = llm or ChatOllama(model=MODEL, temperature=0, base_url=os.getenv("OLLAMA_URL", "http://127.0.0.1:11434"))
    llm_with_tools = llm.bind_tools(TOOLS)

    def assistant(state: MessagesState) -> dict:
        messages = [SystemMessage(SYSTEM_PROMPT)] + state["messages"]
        return {"messages": [llm_with_tools.invoke(messages)]}

    def triage(state: MessagesState) -> dict:
        last = state["messages"][-1].content
        if RED_FLAGS.search(last):
            return {"messages": [AIMessage(MEDICAL_ANSWER, name="triage")]}
        return {}

    def after_triage(state: MessagesState) -> str:
        return END if getattr(state["messages"][-1], "name", None) == "triage" else "assistant"

    def route(state: MessagesState) -> str:
        return "tools" if getattr(state["messages"][-1], "tool_calls", None) else "verify"

    def verify(state: MessagesState) -> dict:
        turn = _current_turn(state["messages"])
        problem = None
        if _needs_new_search(state["messages"], turn):
            problem = ("Пользователь добавил новое условие, а ты предлагаешь товары из старой выдачи. "
                       "Вызови search_catalog заново с учётом ВСЕХ условий диалога.")
        problem = problem or check_answer(state["messages"][-1].content,
                               _allowed_prices(state["messages"], turn),
                               _allowed_actives(turn))
        if problem is None:
            return {}
        retries = sum(1 for m in turn if getattr(m, "name", None) == "verifier")
        if retries >= MAX_RETRIES:
            return {"messages": [AIMessage(FALLBACK, name="fallback")]}
        # Замечание помечено name="verifier": по нему считаем попытки и прячем его из вывода.
        return {"messages": [HumanMessage(f"[Проверка ответа] {problem}", name="verifier")]}

    def after_verify(state: MessagesState) -> str:
        last = state["messages"][-1]
        return "assistant" if getattr(last, "name", None) == "verifier" else END

    graph = StateGraph(MessagesState)
    graph.add_node("triage", triage)
    graph.add_node("assistant", assistant)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_node("verify", verify)
    graph.add_edge(START, "triage")
    graph.add_conditional_edges("triage", after_triage, ["assistant", END])
    graph.add_conditional_edges("assistant", route, ["tools", "verify"])
    graph.add_edge("tools", "assistant")
    graph.add_conditional_edges("verify", after_verify, ["assistant", END])
    # MemorySaver хранит историю диалога по thread_id — так агент помнит прошлые реплики.
    return graph.compile(checkpointer=MemorySaver())


def chat() -> None:
    agent = build_graph()
    config = {"configurable": {"thread_id": "cli"}, "recursion_limit": RECURSION_LIMIT}
    print(f"Beauty-ассистент ({MODEL}). Пустая строка — выход.\n")
    while True:
        user_input = input("Вы: ").strip()
        if not user_input:
            break
        result = agent.invoke({"messages": [HumanMessage(user_input)]}, config)

        # Показываем вызовы инструментов и срабатывания проверки — для отладки и демо.
        messages = result["messages"]
        user_turns = [i for i, m in enumerate(messages) if m.type == "human" and m.name != "verifier"]
        for m in messages[user_turns[-1] + 1:]:
            for call in getattr(m, "tool_calls", None) or []:
                print(f"  [tool] {call['name']}({call['args']})")
            if m.name == "verifier":
                print(f"  [verify] {m.content.removeprefix('[Проверка ответа] ')}")

        print(f"\nАссистент: {messages[-1].content}\n")


if __name__ == "__main__":
    chat()
