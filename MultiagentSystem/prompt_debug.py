"""Run-scoped debug dumps of each LLM agent's input prompt and response.

There is no shared agent base class — every agent is a standalone LangGraph node
that assembles a local ``messages`` list right before ``llm...invoke(messages)``.
This helper gives those nodes one consistent place to persist the exact prompt
(system + human messages + validator feedback) together with the LLM's structured
response, so a whole run can be inspected afterwards for debugging.

Files are written under the run folder threaded into state as ``debug_run_dir``
(created once per run in ``multiagent_predictions_module``):

    <debug_run_dir>/<forecast_date>/<agent_short>[__<sub_label>]__attempt<N>.txt
"""

import json
from datetime import datetime
from pathlib import Path


def _short(agent_name: str) -> str:
    return agent_name.replace("agent_for_analysing_", "").replace("agent_for_", "")


def save_agent_io(
    state,
    agent_name,
    messages,
    *,
    attempt=0,
    forecast_date=None,
    sub_label=None,
    llm_model=None,
    response=None,
    error=None,
):
    """Write one agent's prompt (+ response / error) to the run's debug folder.

    No-op when ``debug_save_prompts`` is off in config or no ``debug_run_dir`` is set
    in state. Intended to be called twice per agent: once before ``invoke`` (prompt
    only, so it survives an LLM failure) and once after with ``response=`` / ``error=``;
    the second call overwrites the same file with the complete record.
    """
    cfg = state.get("config", {}) or {}
    run_dir = state.get("debug_run_dir")
    if not cfg.get("debug_save_prompts", True) or not run_dir:
        return

    forecast_date = forecast_date or state.get("forecast_start_date", "unknown")
    out_dir = Path(run_dir) / str(forecast_date)
    out_dir.mkdir(parents=True, exist_ok=True)

    name = _short(agent_name)
    if sub_label:
        name += f"__{_short(sub_label)}"
    fpath = out_dir / f"{name}__attempt{attempt}.txt"

    lines = [
        "=== AGENT PROMPT DEBUG ===",
        f"agent: {agent_name}",
        f"forecast_date: {forecast_date}",
        f"attempt: {attempt}",
        f"llm_model: {llm_model}",
        f"horizon: {state.get('horizon')}",
        f"saved_at: {datetime.now().isoformat(timespec='seconds')}",
        "",
    ]
    for i, m in enumerate(messages, 1):
        lines += [f"--- MESSAGE {i} [{type(m).__name__}] ---", str(m.content), ""]

    lines.append("=== LLM RESPONSE ===")
    if error is not None:
        lines.append(f"ERROR: {error}")
    elif response is not None:
        try:
            payload = response.model_dump()  # pydantic BaseModel
        except AttributeError:
            payload = response
        lines.append(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        lines.append("(pending)")

    fpath.write_text("\n".join(lines), encoding="utf-8")
