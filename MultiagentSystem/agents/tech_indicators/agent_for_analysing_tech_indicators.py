import json
from pathlib import Path
from typing import Literal, cast

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from llm_factory import make_chat_llm
from multiagent_types import AgentState, get_agent_settings
from pydantic import BaseModel

AGENT_DIR = Path(__file__).parent
AGENT_NAME = "agent_for_analysing_tech_indicators"

# Schema for the LLM's structured output only. Early abstain-branches below
# (data gap, missing close_price, LLM failure) return a plain dict with
# prediction=None / confidence=None, bypassing this schema. Downstream
# (validator, reports_analyser) explicitly skips None votes.
class TechAnalysisResponse(BaseModel):
    reasoning: str        # step-by-step analysis of all indicators following the prompt structure
    summary: str          # brief final conclusion: forecast + confidence + range
    risks: str            # risks and counter-arguments to the forecast (2-5 points or "")
    prediction: bool      # True = HIGHER, False = LOWER (always pick a direction)
    confidence: Literal["high", "medium", "low"]

def agent_for_analysing_tech_indicators(state: AgentState):
    TAG = "[agent_for_analysing_tech_indicators]"

    if AGENT_NAME not in state.get("agent_envolved_in_prediction", []):
        print(f"{TAG} Not in agent_envolved_in_prediction — skipping")
        return {}

    my_retry = None
    for r in state.get("retry_agents", []):
        if r["agent_name"] == AGENT_NAME:
            my_retry = r
            break

    if my_retry is not None and my_retry["currents_retry"] >= my_retry["max_retries"]:
        print(f"{TAG} Retry limit reached ({my_retry['currents_retry']}/{my_retry['max_retries']}) — skipping")
        return {}

    attempt = my_retry["currents_retry"] if my_retry is not None else 0
    max_attempts = my_retry["max_retries"] if my_retry is not None else 2
    print(f"\n{'='*60}")
    print(f"{TAG} === ATTEMPT #{attempt + 1}/{max_attempts} ===")
    print(f"{'='*60}")

    # 1. Get all agent settings
    settings = get_agent_settings(state, "agent_for_analysing_tech_indicators")
    horizon = state["horizon"]
    forecast_date = state["forecast_start_date"]
    llm_model = settings.get("llm_model", "gpt-4o-mini")
    print(f"{TAG} [STEP 1/6] Settings loaded | horizon={horizon}d | forecast_date={forecast_date}")
    print(
        f"{TAG}   window_to_analysis={settings['window_to_analysis']} | "
        f"base_feats count={len(settings['base_feats'])} | llm_model={llm_model}"
    )

    # 2. We should predict values by the forecast_start_date.
    # Dataset source: SharedBaseDataCache (built in
    # multiagent_predictions_module.make_prediction_for_last_N_days).
    df = state["cached_dataset"].copy()
    print(f"{TAG} [STEP 2/6] Cached dataset shape: {df.shape}")
    # We must take only base_feats columns from dataset
    cols = [c for c in settings["base_feats"] if c in df.columns]
    missing_cols = [c for c in settings["base_feats"] if c not in df.columns]
    if missing_cols:
        print(f"{TAG}   WARNING: {len(missing_cols)} features missing from dataset: {missing_cols[:5]}...")
    print(f"{TAG}   Matched {len(cols)}/{len(settings['base_feats'])} features from config")

    # Take the dates before forecast_date (including forecast_date) and take base_feats columns
    df = df.loc[df["date"] <= pd.Timestamp(forecast_date), ["date"] + cols].tail(settings["window_to_analysis"])
    print(f"{TAG}   Filtered data shape: {df.shape} | date range: {df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()}")

    input_csv_path = AGENT_DIR / "input_data.csv"
    df.to_csv(input_csv_path, index=False)
    print(f"{TAG}   Window dataset saved to {input_csv_path.name} ({df.shape[0]} rows x {df.shape[1]} cols)")

    last_date = df["date"].iloc[-1].date() if len(df) > 0 else None
    expected_date = pd.Timestamp(forecast_date).date()
    if last_date != expected_date:
        msg = (
            f"Data ends at {last_date}, expected {expected_date}. "
            f"SharedBaseDataCache may have a data gap."
        )
        print(f"{TAG}   WARNING: {msg} — returning neutral stub signal")
        return {"agent_signals": {AGENT_NAME: {
            "reasoning": msg,
            "summary": "Data gap — abstain from voting.",
            "risks": "",
            "prediction": None,
            "confidence": None,
            "description_of_the_reports_problem": [],
        }}}
    print(f"{TAG}   Last date validated: {last_date} == {forecast_date}")

    # 3. Convert to JSON for the prompt and save for debugging
    data_json = df.to_json(orient="records", date_format="iso")
    (AGENT_DIR / "input_data.json").write_text(data_json, encoding="utf-8")
    print(f"{TAG} [STEP 3/6] Input data saved to input_data.json ({len(data_json)} chars)")

    # 4. Extract current closing price (last row). Close column is configurable
    # (defaults to spot_price_history__close) but remains a hard dependency:
    # if it's not in base_feats the agent abstains on every forecast.
    close_col = settings.get("close_col", "spot_price_history__close")
    if close_col not in settings["base_feats"]:
        print(f"{TAG}   WARNING: close_col '{close_col}' not in base_feats — agent will abstain every day")
    close_price = df[close_col].iloc[-1] if close_col in df.columns else "N/A"
    print(f"{TAG} [STEP 4/6] Close price on {forecast_date}: {close_price}")
    if close_price == "N/A":
        msg = f"Cannot make prediction without close_price for date {forecast_date}"
        print(f"{TAG}   ERROR: {msg}")
        return {"agent_signals": {AGENT_NAME: {
            "reasoning": msg,
            "summary": "Missing close_price — abstain from voting.",
            "risks": "",
            "prediction": None,
            "confidence": None,
            "description_of_the_reports_problem": [],
        }}}

    # 5. Load system prompt (from file or inline)
    if "system_prompt_file" in settings:
        prompt_path = Path(__file__).parent.parent.parent / settings["system_prompt_file"]
        system_prompt = prompt_path.read_text(encoding="utf-8")
        print(f"{TAG} [STEP 5/6] System prompt loaded from file: {settings['system_prompt_file']} ({len(system_prompt)} chars)")
    else:
        system_prompt = settings["system_prompt"]
        print(f"{TAG} [STEP 5/6] System prompt loaded from config ({len(system_prompt)} chars)")
    # Bind prompt instructions to runtime horizon from config/state.
    system_prompt = system_prompt.replace("{HORIZON_DAYS}", str(horizon))

    # 6. Call LLM with CoT: reasoning is filled first, summary is based on it
    # temperature=0 for deterministic backtesting and reduced sampling noise on
    # confidence boundaries (medium/high). Non-zero temperature added ~5-10%
    # vote variance on borderline days.
    llm = make_chat_llm(llm_model, temperature=0)

    # Read full validator-feedback history from retry_agents.retry_requirements
    # (accumulates across iterations) instead of agent_signals.description_of_the_reports_problem
    # (gets wiped by merge_dicts when this agent returns its new signal).
    prev_feedback: list[str] = []
    for r in state.get("retry_agents", []):
        if r["agent_name"] == AGENT_NAME:
            prev_feedback = list(r.get("retry_requirements", []))
            break

    # 6. Build conversation prompt
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=(
            f"Data for {settings['window_to_analysis']} days up to {forecast_date} inclusive:\n{data_json}\n\n"
            f"Current BTC closing price: {close_price}\n"
            f"Forecast horizon: {horizon} days\n"
            f"Forecast date: {forecast_date}\n\n"
            f"Perform analysis following the three steps from the instructions and provide a forecast."
        )),
    ]

    # Add previous feedbacks
    if prev_feedback:
        history_text = "\n".join(
            f"Iteration {i+1}: {d}" for i, d in enumerate(prev_feedback)
        )
        messages.append(HumanMessage(content=(
            f"VALIDATOR FEEDBACK ON PREVIOUS REPORT VERSIONS:\n{history_text}\n\n"
            f"Take this feedback into account when composing the new report."
        )))
        print(f"{TAG}   Including {len(prev_feedback)} previous validator feedback(s)")
    else:
        print(f"{TAG}   No previous validator feedback")

    message_csv_path = AGENT_DIR / "message_to_bot.csv"
    pd.DataFrame(
        [{"role": type(m).__name__, "content": m.content} for m in messages]
    ).to_csv(message_csv_path, index=False)
    print(f"{TAG}   Prompt saved to {message_csv_path.name} ({len(messages)} message(s))")

    print(f"{TAG} [STEP 6/6] Calling LLM ({llm_model}) with {len(messages)} messages...")

    # Tells the LLM to return a JSON object that matches the Pydantic schema, instead of free-form text.
    try:
        response = cast(TechAnalysisResponse, llm.with_structured_output(TechAnalysisResponse).invoke(messages))
    except Exception as exc:
        err = f"LLM request failed in {AGENT_NAME}: {exc}"
        print(f"{TAG}   ERROR: {err}")
        return {"agent_signals": {AGENT_NAME: {
            "reasoning": err,
            "summary": "LLM temporarily unavailable — abstain from voting.",
            "risks": "Network/API issue during model call.",
            "prediction": None,
            "confidence": None,
            "description_of_the_reports_problem": [],
        }}}

    prediction_label = "HIGHER" if response.prediction else "LOWER"
    print(f"{TAG} LLM response received:")
    print(f"{TAG}   Prediction: {prediction_label}")
    print(f"{TAG}   Confidence: {response.confidence}")
    print(f"{TAG}   Reasoning: {response.reasoning[:200]}...")
    print(f"{TAG}   Summary: {response.summary[:200]}")
    print(f"{TAG}   Risks: {response.risks[:200]}")

    print(f"{TAG} Done. Returning signal to graph.")
    return {"agent_signals": {AGENT_NAME: {
        "reasoning": response.reasoning,
        "summary": response.summary,
        "risks": response.risks,
        "prediction": response.prediction,
        "confidence": response.confidence,
        "description_of_the_reports_problem": [],
    }}}
