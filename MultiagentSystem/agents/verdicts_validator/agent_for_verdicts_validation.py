import os
from pathlib import Path
from typing import cast

from langchain_core.messages import HumanMessage, SystemMessage
from llm_factory import make_chat_llm
from multiagent_types import AgentState, AgentSignal, AgentRetry, NON_VALIDATED_AGENTS
from pydantic import BaseModel


class AgentValidationResult(BaseModel):
    has_problem: bool
    description: str        # Specific problem description or "" if no problems


_PROMPT_PATH = Path(__file__).parent / "system_prompt.md"
VALIDATOR_SYSTEM_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8").strip()

# Skip-list for validation — shared with multiagent_graph._NO_RETRY_AGENTS via
# multiagent_types.NON_VALIDATED_AGENTS so the two cannot drift apart.
_SKIP_AGENTS = NON_VALIDATED_AGENTS

TAG = "[validator]"


_DEFAULT_VALIDATOR_MODEL = "claude-sonnet-4-5"


def _resolve_validator_model(state: AgentState) -> str:
    cfg = state.get("config", {}) or {}
    return (
        cfg.get("validator_llm_model")
        or cfg.get("agent_settings", {}).get("verdicts_validator", {}).get("llm_model")
        or _DEFAULT_VALIDATOR_MODEL
    )


def agent_for_verdicts_validation(state: AgentState):
    model = _resolve_validator_model(state)
    if model.startswith("claude") and not os.getenv("CLAUDE_KEY"):
        print(f"{TAG} CLAUDE_KEY env var not set — validator will skip all checks (model={model})")
        return {"retry_agents": [], "agent_signals": {}}
    llm = make_chat_llm(model, temperature=0)

    # Build mutable lookup: agent_name → copy of AgentRetry entry
    retry_map: dict[str, AgentRetry] = {r["agent_name"]: cast(AgentRetry, dict(r)) for r in state.get("retry_agents", [])}

    updated_retry: list[AgentRetry] = []
    updated_signals: dict[str, AgentSignal] = {}

    for agent_name, signal in (state.get("agent_signals") or {}).items():
        # Skip formula-based agents (twitter etc.) — nothing to validate
        if agent_name in _SKIP_AGENTS:
            print(f"{TAG} {agent_name}: skipped (formula-based agent)")
            continue

        # Safety net: skip anything without a retry entry
        if agent_name not in retry_map:
            print(f"{TAG} {agent_name}: skipped (no retry entry)")
            continue

        # Skip agents that returned no reasoning (e.g. skipped on retry)
        if not signal.get("reasoning"):
            print(f"{TAG} {agent_name}: skipped (no reasoning in signal)")
            continue

        prediction = signal.get("prediction")
        # Agent already chose to abstain — nothing to validate. Without this skip
        # the validator would label `None` as "False (LOWER)" and likely flag a
        # bogus retry (data-gap reasoning vs. SHORT vote), wasting LLM budget.
        if prediction is None:
            print(f"{TAG} {agent_name}: skipped (agent abstained — nothing to validate)")
            continue
        prediction_label = "True (HIGHER)" if prediction else "False (LOWER)"

        messages = [
            SystemMessage(content=VALIDATOR_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"Agent: {agent_name}\n"
                f"Reasoning: {signal.get('reasoning', '')}\n"
                f"Summary: {signal.get('summary', '')}\n"
                f"Risks: {signal.get('risks', '')}\n"
                f"Prediction: {prediction_label}\n"
            )),
        ]

        try:
            result = cast(
                AgentValidationResult,
                llm.with_structured_output(AgentValidationResult).invoke(messages),
            )
        except Exception as exc:
            print(f"{TAG} {agent_name}: LLM call failed — {exc}, skipping validation")
            continue

        if result.has_problem:
            entry = retry_map[agent_name]
            entry["retry_requirements"] = list(entry["retry_requirements"]) + [result.description]
            entry["currents_retry"] += 1
            updated_retry.append(entry)

            updated_signal = cast(AgentSignal, dict(signal))
            problems = list(signal.get("description_of_the_reports_problem") or [])
            problems.append(result.description)
            updated_signal["description_of_the_reports_problem"] = problems
            updated_signals[agent_name] = updated_signal

            print(f"{TAG} {agent_name}: PROBLEM (attempt {entry['currents_retry']}/{entry['max_retries']}) — {result.description}")
        else:
            # Clear outstanding retry_requirements for this agent so the router
            # does not re-trigger a retry loop on stale history from a previous
            # failed attempt. currents_retry is preserved as an attempt counter.
            entry = retry_map[agent_name]
            if entry.get("retry_requirements"):
                entry["retry_requirements"] = []
                updated_retry.append(entry)
            print(f"{TAG} {agent_name}: OK")

    return {"retry_agents": updated_retry, "agent_signals": updated_signals}
