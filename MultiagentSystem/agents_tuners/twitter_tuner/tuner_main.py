import json
import sys
from datetime import date, timedelta
from pathlib import Path

from ...agents.twitter_analyser.full_scrapping_pipeline import (
    run_classify_unclassified,
    run_fetch_only,
)
from ...multiagent_graph import app
from .twitter_tuner import find_best_hyperparameters


BASE_DIR = Path(__file__).resolve().parent
TOP_SOURCES_PATH = BASE_DIR / "unbias_top_sources.json"
BASIC_PARAMS_PATH = BASE_DIR / "1_day_basic_parameters.json"
TUNING_OUTPUT_PATH = BASE_DIR / "tuning_top.json"
UNBIAS_BUCKET = "Overall"
UNBIAS_TOP_N = 15


def _load_unique_sources_from_unbias(
    path: Path,
    bucket: str = UNBIAS_BUCKET,
    top_n: int = UNBIAS_TOP_N,
) -> list[str]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    top = payload.get("top_by_accuracy", {})
    rows = top.get(bucket, [])

    seen: set[str] = set()
    handles: list[str] = []
    for item in rows:
        handle = (item.get("handle") or "").strip().lstrip("@")
        if handle and handle not in seen:
            seen.add(handle)
            handles.append(handle)
        if len(handles) >= top_n:
            break
    return handles


def _load_basic_params(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> list[dict]:
    if not TOP_SOURCES_PATH.exists():
        raise FileNotFoundError(f"Missing file: {TOP_SOURCES_PATH}")
    if not BASIC_PARAMS_PATH.exists():
        raise FileNotFoundError(f"Missing file: {BASIC_PARAMS_PATH}")

    authors = _load_unique_sources_from_unbias(TOP_SOURCES_PATH)
    if not authors:
        raise ValueError("No sources found in unbias_top_sources.json")

    params = _load_basic_params(BASIC_PARAMS_PATH)

    today = date.today()
    forecast_date = today.strftime("%Y-%m-%d")
    since_date = (today - timedelta(days=120)).strftime("%Y-%m-%d")
    until_date = forecast_date
    horizon = int(params.get("horizon"))

    print(
        f"[tuner_main] sources from unbias bucket={UNBIAS_BUCKET} top_n={UNBIAS_TOP_N}: {len(authors)}"
    )
    print(f"[tuner_main] tweets window: {since_date} -> {until_date} (100 days)")

    fetch_result = run_fetch_only(
        since_date=since_date,
        until_date=until_date,
        authors=authors,
        stop_on_existing_duplicates=True,
        duplicates_threshold=5
    )
    print(f"[tuner_main] fetch result: {fetch_result}")

    classify_result = run_classify_unclassified(
        since_date=since_date,
        until_date=until_date,
        authors=authors,
    )
    print(f"[tuner_main] classify result: {classify_result}")

    top = find_best_hyperparameters(
        app=app,
        authors=authors,
        forecast_date=forecast_date,
        horizon=horizon,
        save_path=TUNING_OUTPUT_PATH,
        step=float(params.get("step")),
        decay_rate_range=tuple(params.get("decay_rate_range")),
        decay_start_day_range=tuple(params.get("decay_start_day_range")),
        initial_weight_range=tuple(params.get("initial_weight_range")),
        window_to_analysis_range=tuple(params.get("window_to_analysis_range")),
        eval_days=int(params.get("eval_days")),
        top_k=int(params.get("top_k")),
        balance_penalty_weight=float(params.get("balance_penalty_weight")),
        min_accuracy=float(params.get("min_accuracy")),
        max_gap=float(params.get("max_gap")),
    )

    print(f"[tuner_main] tuning finished. Saved: {TUNING_OUTPUT_PATH}")
    return top


if __name__ == "__main__":
    log_path = BASE_DIR / "logs.log"
    sys.stdout = open(str(log_path), "w", encoding="utf-8", buffering=1)
    sys.stderr = sys.stdout

    main()
