from multiagent_types import AgentState


CONFIDENCE_WEIGHTS = {"high": 3, "medium": 2, "low": 1}


def compute_confidence_score(
    agent_signals: dict,
    neutral_threshold: float = 1.0,
) -> tuple[float, str | None, str]:
    """
    Mathematical aggregation of agent predictions on the same scale as a single
    agent vote, i.e. score in [-3, +3].

    Each agent contributes sign(prediction) * weight, where
        weight  = CONFIDENCE_WEIGHTS[confidence]  (low=1, medium=2, high=3)
        sign    = +1 if prediction is True (HIGHER), else -1 (LOWER)

    The final score is the arithmetic mean over all *real* (non-stub) agents,
    so adding more agents does not inflate the magnitude.

    Returns (score, direction, breakdown_text).
    - score: float in [-3, 3]
    - direction: "LONG" | "SHORT" | None (None inside the neutral band)
    - breakdown_text: text breakdown of the calculation
    """
    
    # Agent votes
    agents_votes: list[float] = []
    # For general logs about predictions from agents
    parts: list[str] = []

    for name, signal in agent_signals.items():
        prediction = signal.get("prediction")
        confidence = signal.get("confidence")

        # Skip signals that do not carry a vote (stub agents, or formula-based
        # agents that returned "no actionable signal" for this forecast date).
        if prediction is None or confidence is None:
            parts.append(f"{name}: no vote (skipped)")
            continue

        weight = CONFIDENCE_WEIGHTS.get(str(confidence).strip().lower(), 1)
        sign = 1 if prediction is True else -1
        vote = sign * weight

        agents_votes.append(vote)
        label = "HIGHER" if sign > 0 else "LOWER"
        parts.append(f"{name}: {label} ({confidence}) -> {vote:+d}")

    if not agents_votes:
        return 0.0, None, "No real reports"

    score = sum(agents_votes) / len(agents_votes)

    # GENERAL VERDICT
    breakdown = (
        " | ".join(parts)
        + f" => mean({len(agents_votes)} agents) = {score:+.2f}"
    )

    if score > neutral_threshold:
        direction = "LONG"
    elif score < -neutral_threshold:
        direction = "SHORT"
    else:
        direction = None

    return score, direction, breakdown


def agent_reports_analyser(state: AgentState):
    TAG = "[reports_analyser]"
    signals = state.get("agent_signals", {})
    threshold = state.get("config", {}).get("neutral_threshold", 1)

    print(f"\n{'='*60}")
    print(f"{TAG} === AGGREGATING FINAL VERDICT ===")
    print(f"{'='*60}")
    print(f"{TAG} Received {len(signals)} agent signals | neutral_threshold={threshold}")

    for name, signal in signals.items():
        pred = signal.get("prediction")
        conf = signal.get("confidence", "?")
        reasoning = signal.get("reasoning") or "(empty)"
        summary = signal.get("summary") or "(empty)"
        risks = signal.get("risks") or "(empty)"
        problems = signal.get("description_of_the_reports_problem", [])
        pred_label = "HIGHER" if pred is True else ("LOWER" if pred is False else "N/A")

        print(f"{TAG}   --- {name} ---")
        print(f"{TAG}     prediction: {pred_label}")
        print(f"{TAG}     confidence: {conf}")
        print(f"{TAG}     validation_issues: {len(problems)}")
        print(f"{TAG}     summary:   {summary}")
        print(f"{TAG}     reasoning: {reasoning}")
        print(f"{TAG}     risks:     {risks}")

    score, direction, breakdown = compute_confidence_score(signals, threshold)

    direction_label = direction or "NEUTRAL"
    print(f"{TAG} Score calculation: {breakdown}")
    print(f"{TAG} Final score: {score:+.2f} in [-3, +3] (neutral zone: +/-{threshold}) -> {direction_label}")

    reasoning = f"Confidence score: {score:+.2f} in [-3, +3] (neutral zone: +/-{threshold}). {breakdown}"
    summary = f"{direction_label} (score={score:+.2f})"

    print(f"{TAG} Done. Final verdict: {direction_label}")

    return {
        "general_prediction_by_all_reports": direction,
        "general_reports_summary": summary,
        "general_reports_reasoning": reasoning,
        "general_reports_risks": "",
        "confidence_score": score,
    }
