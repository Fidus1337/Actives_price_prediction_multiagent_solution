from datetime import date
from typing import List, Literal, Union, Annotated

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from typing_extensions import TypedDict


# Reducer для agent_signals — мержит словари по ключу
def merge_dicts(left: dict, right: dict) -> dict:
    if not left: left = {}
    if not right: right = {}
    return {**left, **right}


# Reducer для retry_agents — заменяет AgentRetry по agent_name, добавляет новые
def merge_retry_agents(left: list, right: list) -> list:
    if not left: left = []
    if not right: right = []
    # Берём существующий список и заменяем/добавляем элементы из right по agent_name
    result = {r["agent_name"]: r for r in left}
    for r in right:
        result[r["agent_name"]] = r  # перезаписывает старый если agent_name совпадает
    return list(result.values())

# This is class, which exists for structuring reports by agents
class AgentSignal(TypedDict):
    description_of_the_reports_problem: list[str]
    reasoning: str
    summary: str
    risks: str        # contrarguments about end prediction
    prediction: bool  # True(up)/False(down)
    confidence: str   # high / medium / low

class AgentRetry(TypedDict):
    agent_name: str
    max_retries: int
    currents_retry: int
    retry_requirements: list[str]

# General agent state
class AgentState(TypedDict):
    config: dict # config from MultiagentSystem folder, all settings for launching precitions
    agent_envolved_in_prediction: list[str]
    cached_dataset: pd.DataFrame | None  # SharedBaseDataCache base_df
    horizon: int
    general_prediction_by_all_reports: Literal["LONG", "SHORT"] | None # after analysis by agent_reports_analyser, we can skip predicts
    general_reports_summary: str
    general_reports_reasoning: str
    general_reports_risks: str
    confidence_score: float  # in [-3, +3], sign indicates direction, |score| indicates strength
    forecast_start_date: str
    save_results: bool                 # write each prediction row to CSV at save_path
    save_path: str | None              # CSV path; ignored when save_results is False
    agent_signals: Annotated[dict[str, AgentSignal], merge_dicts] # every agent returns signal
    retry_agents: Annotated[list[AgentRetry], merge_retry_agents]

def get_agent_settings(state: AgentState, agent_name: str) -> dict:
    """Get settings for a specific agent from state."""
    return state["config"]["agent_settings"][agent_name]


# Single source of truth for which agents are NOT validated AND NOT retry-tracked.
# Imported by both multiagent_graph._NO_RETRY_AGENTS and verdicts_validator._SKIP_AGENTS,
# so the two sets cannot drift out of sync.
#
# Excluded agents:
# - agent_for_twitter_analysis: formula-based, no LLM report to validate.
# - agent_for_news_analysis: pre-classified articles, no LLM report to validate.
# - agent_for_economic_calendar_analysis: LLM-based but deterministic at temp=0;
#   validator prompt is written for indicator-based reasoning and doesn't apply.
NON_VALIDATED_AGENTS = frozenset({
    "agent_for_twitter_analysis",
    "agent_for_news_analysis"
})
