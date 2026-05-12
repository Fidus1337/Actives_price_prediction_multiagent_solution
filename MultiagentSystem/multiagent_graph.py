import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph.graph import StateGraph, START, END

from multiagent_types import AgentState, AgentRetry, NON_VALIDATED_AGENTS

from .agents.twitter_analyser.agent_for_twitter_analysis import agent_for_twitter_analysis
from .agents.tech_indicators import agent_for_analysing_tech_indicators
from .agents.onchain_indicators import agent_for_analysing_onchain_indicators
from .agents.news_analyser.agent_for_news_analysis import agent_for_news_analysis
from .agents.economic_calendar_analyser import agent_for_economic_calendar_analysis
from .agents.verdicts_validator import agent_for_verdicts_validation
from .agents.reports_analyser import agent_reports_analyser

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# Agents excluded from retry tracking — must equal validator's skip-set, hence
# both come from multiagent_types.NON_VALIDATED_AGENTS. An agent that isn't
# validated cannot accumulate retry_requirements, so giving it a retry-entry is
# dead weight. Retry-eligible agents: tech indicators, onchain indicators.
_NO_RETRY_AGENTS = NON_VALIDATED_AGENTS

# Max total attempts per agent, INCLUDING the first run — not additional
# retries on top of it. MAX_RETRIES=2 means: first run, then at most one
# retry with validator feedback. Name kept for compat with AgentRetry.max_retries.
MAX_RETRIES = 2


# Node for starting analysis by agents
def supervisor_node(state: AgentState):
    retry_agents: list[AgentRetry] = state.get("retry_agents", [])

    if not retry_agents:
        # First run: initialize retry tracking for every non-news agent involved
        involved: list[str] = state.get("agent_envolved_in_prediction", [])
        initialized: list[AgentRetry] = [
            AgentRetry(
                agent_name=name,
                max_retries=MAX_RETRIES,
                currents_retry=0,
                retry_requirements=[],
            )
            for name in involved
            if name not in _NO_RETRY_AGENTS
        ]
        names = [r["agent_name"] for r in initialized]
        print(f"\n[supervisor] First run — retry tracking initialized for: {names}")
        return {"retry_agents": initialized}

    names = [r["agent_name"] for r in retry_agents if r.get("retry_requirements")]
    print(f"\n[supervisor] Retry run — agents with requirements: {names}")
    return {}


# Router for retry (look schema at miro)
def _should_retry(state: AgentState) -> str:
    retry_agents: list[AgentRetry] = state.get("retry_agents", [])

    # If the agent has any problems - retry
    agents_with_budget = [
        r for r in retry_agents
        # If report still has error and we have additional retries
        if r.get("retry_requirements") and len(r["retry_requirements"]) < r["max_retries"]
    ]

    if agents_with_budget:
        names = [r["agent_name"] for r in agents_with_budget]
        print(f"\n[router] Agents need retry: {names}")
        return "supervisor"

    exhausted = [
        r["agent_name"] for r in retry_agents
        if r.get("retry_requirements") and len(r["retry_requirements"]) >= r["max_retries"]
    ]
    if exhausted:
        print(f"\n[router] Retry limit ({MAX_RETRIES}) exhausted for: {exhausted}")
    else:
        print(f"\n[router] All agents passed validation — finishing.")

    return "agent_reports_analyser"


def build_multiagent_graph():
    """Build and compile the multiagent LangGraph DAG."""
    # ==========================================
    # STEP 1: INITIALIZATION AND ADDING NODES
    # ==========================================
    builder = StateGraph(AgentState)

    # Register all nodes (node names are defined as strings)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("agent_for_analysing_tech_indicators", agent_for_analysing_tech_indicators)
    builder.add_node("agent_for_analysing_onchain_indicators", agent_for_analysing_onchain_indicators)
    builder.add_node("agent_for_news_analysis", agent_for_news_analysis)
    builder.add_node("agent_for_twitter_analysis", agent_for_twitter_analysis)
    builder.add_node("agent_for_economic_calendar_analysis", agent_for_economic_calendar_analysis)
    builder.add_node("validator", agent_for_verdicts_validation)
    builder.add_node("agent_reports_analyser", agent_reports_analyser)

    # ==========================================
    # STEP 2: BUILDING EDGES (ROUTING)
    # ==========================================
    # 1. Entry point: from system start go to supervisor
    builder.add_edge(START, "supervisor")

    # 2. PARALLEL BRANCHING (Fan-out)
    # Draw edges from supervisor to agents.
    # LangGraph will detect this and run them simultaneously!
    builder.add_edge("supervisor", "agent_for_analysing_tech_indicators")
    builder.add_edge("supervisor", "agent_for_analysing_onchain_indicators")
    builder.add_edge("supervisor", "agent_for_news_analysis")
    builder.add_edge("supervisor", "agent_for_twitter_analysis")
    builder.add_edge("supervisor", "agent_for_economic_calendar_analysis")

    # 3. MERGE (Fan-in)
    # The array means: "Wait for all these nodes to complete,
    # and only then pass control to validator"
    builder.add_edge(
        [
            "agent_for_twitter_analysis",
            "agent_for_analysing_tech_indicators",
            "agent_for_analysing_onchain_indicators",
            "agent_for_news_analysis",
            "agent_for_economic_calendar_analysis",
        ],
        "validator",
    )

    # 4. Conditional exit: if there are agents with recompose_report=True — retry from supervisor
    builder.add_conditional_edges("validator", _should_retry)

    builder.add_edge("agent_reports_analyser", END)

    # ==========================================
    # STEP 3: GRAPH COMPILATION
    # ==========================================
    return builder.compile()


app = build_multiagent_graph()
