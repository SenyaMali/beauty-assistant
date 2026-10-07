"""Оценка агента целиком: диалоги из agent_cases.json и автоматические проверки.

Каждый кейс — новый диалог (отдельный thread_id). Проверяется последний ход.

Проверки из кейса:
  clarifies        — агент задал уточняющий вопрос и не искал;
  filters          — фильтры, с которыми реально прошёл поиск (filters_used из инструмента).
                     null означает «фильтра быть не должно»;
  recommends       — назван хотя бы один товар из выдачи;
  no_products      — не назван ни один товар;
  must_name_ids    — названы конкретные товары;
  must_mention_any — в ответе есть одно из слов;
  tool_called      — вызван инструмент; no_tool — инструменты не вызывались.

Проверки для всех кейсов (сквозные):
  constraints — каждый названный товар удовлетворяет фильтрам поиска
                (цена ≤ бюджета, нужный тип кожи, актив и т.д.);
  no_fallback — агент не сдался (verify не исчерпал попытки).

Отдельно считаются, но на прохождение не влияют:
  model_filters — какую долю ожидаемых фильтров модель передала САМА, без
                  автодополнения кодом. Показывает, насколько держит LLM, а насколько страхует код;
  verify_retries — сколько раз сработала проверка ответа;
  latency       — секунд на ход.

Запуск (нужен запущенный Ollama):
    python -m eval.run_agent_eval
    OLLAMA_MODEL=qwen2.5:14b python -m eval.run_agent_eval
    python -m eval.run_agent_eval --only A03,A04      # отдельные кейсы
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage, ToolMessage

import re

from app.agent import MODEL, PRICE, RECURSION_LIMIT, build_graph
from app.ingest import CATALOG

CASES = Path(__file__).parent / "agent_cases.json"
RESULTS = Path(__file__).parent / "results"


def _short_name(product: dict) -> str:
    """«Sela Pure пенка для умывания» → «sela pure»: бренд + слово серии, так модель пишет чаще всего."""
    return " ".join(product["name"].lower().split()[:-len(product["category"].split())])


def run_case(agent, case: dict) -> dict:
    config = {"configurable": {"thread_id": case["id"]}, "recursion_limit": RECURSION_LIMIT}
    latencies = []
    for text in case["turns"]:
        t0 = time.perf_counter()
        result = agent.invoke({"messages": [HumanMessage(text)]}, config)
        latencies.append(time.perf_counter() - t0)

    messages = result["messages"]
    last_user = max(i for i, m in enumerate(messages) if m.type == "human" and m.name != "verifier")
    turn = messages[last_user + 1:]
    answer = messages[-1].content

    tool_calls = [c for m in turn for c in (getattr(m, "tool_calls", None) or [])]
    search_outputs = []
    found = {}  # id -> товар из выдачи
    # Если в последнем ходе поиска не было, ответ опирается на выдачу прошлого хода —
    # оцениваем по ней (последний поиск в диалоге).
    searched_now = any(isinstance(m, ToolMessage) for m in turn)
    scope = turn if searched_now else messages
    for m in scope:
        if isinstance(m, ToolMessage) and m.name == "search_catalog":
            try:
                data = json.loads(m.content)
            except json.JSONDecodeError:
                search_outputs.append({"empty": True, "text": m.content})
                continue
            search_outputs.append(data)
            found.update({p["id"]: p for p in data["products"]})
        elif isinstance(m, ToolMessage) and m.name == "get_product":
            try:
                p = json.loads(m.content)
                found[p["id"]] = p
            except (json.JSONDecodeError, KeyError):
                pass

    low = answer.lower()
    named = [pid for pid, p in found.items() if _short_name(p) and _short_name(p) in low]
    used = next((s["filters_used"] for s in reversed(search_outputs) if "filters_used" in s), None)
    model_args = next((c["args"] for c in reversed(tool_calls) if c["name"] == "search_catalog"), {})

    checks = case["checks"]
    results: dict[str, bool] = {}

    if checks.get("clarifies"):
        results["clarifies"] = not tool_calls and "?" in answer
    if "filters" in checks:
        if used is None:
            results["filters"] = False
        else:
            results["filters"] = all(used.get(k) == v for k, v in checks["filters"].items())
    if checks.get("recommends"):
        results["recommends"] = bool(named)
    if checks.get("no_products"):
        # Цены в ответе допустимы, только если это бюджет, названный пользователем.
        budgets = {int(n) for t in case["turns"] for n in re.findall(r"\d{3,6}", t.replace(" ", ""))}
        prices = {int(re.sub(r"\D", "", x)) for x in PRICE.findall(answer)}
        results["no_products"] = not named and not (prices - budgets)
    if "must_name_ids" in checks:
        results["must_name_ids"] = all(pid in named for pid in checks["must_name_ids"])
    if "must_mention_any" in checks:
        results["must_mention_any"] = any(w in low for w in checks["must_mention_any"])
    if "tool_called" in checks:
        results["tool_called"] = any(c["name"] == checks["tool_called"] for c in tool_calls)
    if checks.get("no_tool"):
        results["no_tool"] = not tool_calls

    # Сквозные проверки
    if used and named:
        def ok(p: dict) -> bool:
            return all([
                not used.get("skin_type") or used["skin_type"] in p["skin_types"],
                not used.get("category") or used["category"] == p["category"],
                not used.get("active") or used["active"] in p["actives"],
                not used.get("max_price") or p["price_rub"] <= used["max_price"],
                not used.get("fragrance_free") or p["fragrance_free"],
            ])
        results["constraints"] = all(ok(found[pid]) for pid in named)
    results["no_fallback"] = messages[-1].name != "fallback"

    expected_filters = {k: v for k, v in checks.get("filters", {}).items() if v is not None}
    model_hits = sum(1 for k, v in expected_filters.items() if model_args.get(k) == v)

    return {
        "id": case["id"], "name": case["name"],
        "passed": all(results.values()),
        "checks": results,
        "model_filters": (model_hits, len(expected_filters)),
        "verify_retries": sum(1 for m in turn if m.name == "verifier"),
        "latency_sec": round(sum(latencies) / len(latencies), 1),
        "filters_used": used, "model_args": model_args,
        "named": named, "answer": answer,
        # Что забраковал verify и почему — без этого не понять, кто виноват: модель или проверка.
        "rejected": [
            {"answer": prev.content, "reason": m.content.removeprefix("[Проверка ответа] ")}
            for prev, m in zip(turn, turn[1:]) if m.name == "verifier"
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="id кейсов через запятую")
    args = parser.parse_args()

    cases = json.loads(CASES.read_text(encoding="utf-8"))
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in wanted]

    agent = build_graph()
    rows = []
    for case in cases:
        print(f"→ {case['id']} {case['name']} ...", end=" ", flush=True)
        try:
            row = run_case(agent, case)
        except Exception as e:  # падение агента — тоже результат, а не повод остановить прогон
            row = {"id": case["id"], "name": case["name"], "passed": False,
                   "checks": {"error": False}, "error": repr(e),
                   "model_filters": (0, 0), "verify_retries": 0, "latency_sec": 0}
        rows.append(row)
        failed = [k for k, v in row["checks"].items() if not v]
        print("OK" if row["passed"] else f"FAIL: {', '.join(failed)}")

    n = len(rows)
    passed = sum(r["passed"] for r in rows)
    mf_hit = sum(r["model_filters"][0] for r in rows)
    mf_all = sum(r["model_filters"][1] for r in rows)
    summary = {
        "model": MODEL,
        "pass_rate": round(passed / n, 3),
        "model_filter_accuracy": round(mf_hit / mf_all, 3) if mf_all else None,
        "verify_retries_total": sum(r["verify_retries"] for r in rows),
        "fallbacks": sum(1 for r in rows if r["checks"].get("no_fallback") is False),
        "avg_latency_sec": round(sum(r["latency_sec"] for r in rows) / n, 1),
    }

    print("\n" + "=" * 60)
    print(f"Модель:                    {summary['model']}")
    print(f"Пройдено кейсов:           {passed}/{n} ({summary['pass_rate']:.0%})")
    if mf_all:
        print(f"Фильтры передала сама LLM: {mf_hit}/{mf_all} ({summary['model_filter_accuracy']:.0%})")
    print(f"Срабатываний verify:       {summary['verify_retries_total']}")
    print(f"Сдался (fallback):         {summary['fallbacks']}")
    print(f"Средняя задержка хода:     {summary['avg_latency_sec']} с")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"agent_{MODEL.replace(':', '-')}_{datetime.now():%Y%m%d_%H%M}.json"
    out.write_text(json.dumps({"summary": summary, "cases": rows}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\nОтветы и детали по каждому кейсу: {out}")


if __name__ == "__main__":
    main()
