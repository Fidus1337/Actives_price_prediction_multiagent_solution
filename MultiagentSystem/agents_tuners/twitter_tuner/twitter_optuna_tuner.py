import copy
import json
from datetime import datetime, timedelta
from pathlib import Path
import sys
import math

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from ...multiagent_predictions_module import add_y_true, make_one_prediction

try:
    import optuna  # type: ignore[reportMissingImports]
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Optuna is required for twitter_optuna_tuner. Install it with `pip install optuna`."
    ) from exc


def _unpack_range(spec, default_step: float) -> tuple[float, float, float]:
    spec = tuple(spec)
    if len(spec) == 2:
        return float(spec[0]), float(spec[1]), float(default_step)
    if len(spec) == 3:
        return float(spec[0]), float(spec[1]), float(spec[2])
    raise ValueError(f"range must be (low, high) or (low, high, step), got {spec}")


def _build_float_grid(spec, default_step: float) -> list[float]:
    low, high, step = _unpack_range(spec, default_step)
    if step <= 0:
        raise ValueError(f"step must be > 0, got {step}")
    if high < low:
        raise ValueError(f"range ({low}, {high}) is empty")
    values = np.arange(low, high + step / 2, step)
    return [round(float(v), 6) for v in values]


def _build_int_grid(spec, default_step: float) -> list[int]:
    low, high, step = _unpack_range(spec, default_step)
    int_step = max(1, round(step * 10)) if step < 1 else max(1, round(step))
    lo, hi = int(round(low)), int(round(high))
    if hi < lo:
        raise ValueError(f"range ({low}, {high}) is empty")
    return list(range(lo, hi + 1, int_step))


def _total_possible_combinations(
    n_authors: int,
    n_windows: int,
    n_decay_rates: int,
    n_decay_start_days: int,
    n_initial_weights: int,
) -> int:
    if n_authors <= 0:
        return 0
    subsets = (2 ** n_authors) - 1
    return subsets * n_windows * n_decay_rates * n_decay_start_days * n_initial_weights


def _wilson_lcb(accuracy: float, n: int, z: float = 1.96) -> float:
    """Wilson lower confidence bound for Bernoulli accuracy."""
    if n <= 0:
        return 0.0
    p = min(max(float(accuracy), 0.0), 1.0)
    denom = 1.0 + (z * z) / n
    center = p + (z * z) / (2.0 * n)
    margin = z * math.sqrt((p * (1.0 - p) / n) + (z * z) / (4.0 * n * n))
    return max(0.0, min(1.0, (center - margin) / denom))


def _ranked_top(
    current_results: list[dict],
    top_k: int,
    min_accuracy: float,
    max_gap: float,
    min_n_valid: int,
) -> list[dict]:
    eligible = [
        r for r in current_results
        if r["accuracy"] >= min_accuracy and r["balance_gap"] <= max_gap
        and r["n_valid"] >= min_n_valid
    ]
    ranked = sorted(
        eligible,
        key=lambda r: (r["accuracy_lcb"], r["n_valid"], r["score"], r["accuracy"]),
        reverse=True,
    )
    unique_ranked: list[dict] = []
    seen: set[tuple] = set()
    for row in ranked:
        hp = row["hyperparameters"]
        sig = (
            tuple(hp["authors"]),
            int(hp["window_to_analysis"]),
            float(hp["decay_rate"]),
            int(hp["decay_start_day"]),
            float(hp["initial_weight"]),
        )
        if sig in seen:
            continue
        seen.add(sig)
        unique_ranked.append(row)
        if len(unique_ranked) >= top_k:
            break
    return [{"rank": i + 1, **r} for i, r in enumerate(unique_ranked)]


def _pareto_front(
    current_results: list[dict],
    min_accuracy: float,
    max_gap: float,
    min_n_valid: int,
) -> list[dict]:
    """Return non-dominated points for objectives (score, n_valid), maximize both."""
    eligible = [
        r for r in current_results
        if r["accuracy"] >= min_accuracy and r["balance_gap"] <= max_gap
        and r["n_valid"] >= min_n_valid
    ]
    pareto: list[dict] = []
    for i, a in enumerate(eligible):
        dominated = False
        for j, b in enumerate(eligible):
            if i == j:
                continue
            better_or_equal = (
                b["score"] >= a["score"]
                and b["n_valid"] >= a["n_valid"]
            )
            strictly_better = (
                b["score"] > a["score"]
                or b["n_valid"] > a["n_valid"]
            )
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            pareto.append(a)
    return pareto


def find_best_hyperparameters_optuna(
    app,
    authors: list[str],
    forecast_date: str | None = None,
    horizon: int | None = None,
    save_path: str | Path | None = None,
    n_trials: int = 500,
    step: float = 0.1,
    decay_rate_range: tuple = (0.05, 0.35, 0.05),
    decay_start_day_range: tuple = (3.0, 14.0, 1),
    initial_weight_range: tuple = (0.5, 1.5, 0.1),
    window_to_analysis_range: tuple = (7, 28, 1),
    eval_days: int = 100,
    top_k: int = 50,
    balance_penalty_weight: float = 1.0,
    min_accuracy: float = 0.57,
    max_gap: float = 0.3,
    min_n_valid: int = 70,
    report_every: int = 20,
    sampler_seed: int = 42,
) -> list[dict]:
    """Tune twitter-agent hyperparameters with Optuna (TPE sampler).

    Search space:
    - Binary include/exclude flag for each author from `authors`
    - `window_to_analysis`, `decay_rate`, `decay_start_day`, `initial_weight`
      sampled from discretized grids built from provided ranges.

    Ranking metric (same as grid tuner):
        score = accuracy - balance_penalty_weight * |TPR - TNR|
    """
    uniq_authors = list(dict.fromkeys(authors))
    if not uniq_authors:
        raise ValueError("authors pool must be non-empty")

    config_path = Path(__file__).resolve().parent.parent.parent.parent / "configs" / "multiagent_config.json"
    with open(config_path, encoding="utf-8") as f:
        base_config = json.load(f)

    if forecast_date is None:
        forecast_date = base_config["forecast_start_date"]
    if horizon is None:
        horizon = int(base_config["horizon"])
    else:
        horizon = int(horizon)

    end_date = datetime.strptime(forecast_date, "%Y-%m-%d")
    forecast_dates = [
        (end_date - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(eval_days)
    ]

    decay_rate_grid = _build_float_grid(decay_rate_range, default_step=step)
    initial_weight_grid = _build_float_grid(initial_weight_range, default_step=step)
    decay_start_day_grid = _build_int_grid(decay_start_day_range, default_step=step)
    window_grid = _build_int_grid(window_to_analysis_range, default_step=step)

    total_possible = _total_possible_combinations(
        n_authors=len(uniq_authors),
        n_windows=len(window_grid),
        n_decay_rates=len(decay_rate_grid),
        n_decay_start_days=len(decay_start_day_grid),
        n_initial_weights=len(initial_weight_grid),
    )
    if total_possible <= 0:
        raise ValueError("Empty search space")

    effective_trials = int(min(max(1, n_trials), total_possible))
    print(
        f"[optuna_tuner] horizon={horizon}d eval_days={eval_days} n_trials={effective_trials} "
        f"total_possible={total_possible} (authors={len(uniq_authors)})"
    )

    load_dotenv(Path(__file__).resolve().parent.parent.parent.parent / "dev.env")
    cached_dataset = None
    print("[optuna_tuner] Twitter-only mode — skipping CoinGlass dataset fetch")

    if save_path is None:
        save_path = Path(__file__).resolve().parent / "tuning_top_optuna.json"
    else:
        save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    evaluated_cache: dict[tuple, dict] = {}

    def _write_payload(top_list: list[dict], trial_no: int, done: bool) -> None:
        pareto = _pareto_front(
            current_results=results,
            min_accuracy=min_accuracy,
            max_gap=max_gap,
            min_n_valid=min_n_valid,
        )
        payload = {
            "method": "optuna_nsga2_multiobjective",
            "objectives": ["score_max", "n_valid_max"],
            "horizon": horizon,
            "eval_days": eval_days,
            "step": step,
            "balance_penalty_weight": balance_penalty_weight,
            "min_accuracy": min_accuracy,
            "max_gap": max_gap,
            "min_n_valid": int(min_n_valid),
            "total_possible_trials": total_possible,
            "requested_trials": int(n_trials),
            "total_trials": int(effective_trials),
            "trials_completed": int(trial_no),
            "done": done,
            "pareto_size": len(pareto),
            "top": top_list,
        }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def _report(trial_no: int, done: bool) -> list[dict]:
        pareto = _pareto_front(
            current_results=results,
            min_accuracy=min_accuracy,
            max_gap=max_gap,
            min_n_valid=min_n_valid,
        )
        ranked = sorted(
            pareto,
            key=lambda r: (r["accuracy_lcb"], r["n_valid"], r["score"], r["accuracy"]),
            reverse=True,
        )[:top_k]
        top_list = [{"rank": i + 1, **r} for i, r in enumerate(ranked)]
        print(
            f"\n[optuna_tuner] === running top-{len(top_list)} after "
            f"{trial_no}/{effective_trials} trials | pareto={len(pareto)} ==="
        )
        for r in top_list:
            hp = r["hyperparameters"]
            print(
                f"  #{r['rank']:<3} score={r['score']:.3f} acc={r['accuracy']:.3f} "
                f"acc_lcb={r['accuracy_lcb']:.3f} "
                f"tpr={r['tpr']:.3f} tnr={r['tnr']:.3f} gap={r['balance_gap']:.3f} "
                f"n_valid={r['n_valid']} "
                f"win={hp['window_to_analysis']} dr={hp['decay_rate']} "
                f"dsd={hp['decay_start_day']} iw={hp['initial_weight']} "
                f"authors={hp['authors']}"
            )
        _write_payload(top_list, trial_no=trial_no, done=done)
        print(f"[optuna_tuner] === end running top (saved → {save_path}) ===\n")
        return top_list

    def _objective(trial: "optuna.trial.Trial") -> tuple[float, float]:
        selected_authors = [
            author
            for author in uniq_authors
            if trial.suggest_categorical(f"use_author__{author}", [False, True])
        ]
        if not selected_authors:
            # Penalize empty subset and keep study moving.
            trial.set_user_attr("skip_reason", "empty_authors_subset")
            return -1.0, 0.0

        window = int(trial.suggest_categorical("window_to_analysis", window_grid))
        decay_rate = float(trial.suggest_categorical("decay_rate", decay_rate_grid))
        decay_start_day = int(trial.suggest_categorical("decay_start_day", decay_start_day_grid))
        initial_weight = float(trial.suggest_categorical("initial_weight", initial_weight_grid))
        signature = (
            tuple(selected_authors),
            int(window),
            float(decay_rate),
            int(decay_start_day),
            float(initial_weight),
        )
        cached = evaluated_cache.get(signature)
        if cached is not None:
            trial.set_user_attr("cached_reuse", True)
            trial.set_user_attr("accuracy", cached["accuracy"])
            trial.set_user_attr("accuracy_lcb", cached["accuracy_lcb"])
            trial.set_user_attr("balance_gap", cached["balance_gap"])
            trial.set_user_attr("n_valid", cached["n_valid"])
            trial.set_user_attr("authors", list(selected_authors))
            return float(cached["score"]), float(cached["n_valid"])

        trial_config = copy.deepcopy(base_config)
        trial_config["forecast_start_date"] = forecast_date
        trial_config["horizon"] = horizon
        trial_config["agent_envolved_in_prediction"] = [
            "agent_for_twitter_analysis"
        ]
        trial_config["agent_settings"]["agent_for_twitter_analysis"] = {
            "window_to_analysis": window,
            "decay_rate": decay_rate,
            "decay_start_day": decay_start_day,
            "initial_weight": initial_weight,
            "authors": selected_authors,
        }

        rows = [
            make_one_prediction(app, trial_config, date, cached_dataset)
            for date in forecast_dates
        ]
        df = add_y_true(pd.DataFrame(rows), horizon)
        valid = df[
            df["y_predict"].isin(["LONG", "SHORT"])
            & df["y_true"].isin(["LONG", "SHORT"])
        ]
        n_valid = len(valid)
        tp = int(((valid["y_predict"] == "LONG") & (valid["y_true"] == "LONG")).sum())
        tn = int(((valid["y_predict"] == "SHORT") & (valid["y_true"] == "SHORT")).sum())
        fp = int(((valid["y_predict"] == "LONG") & (valid["y_true"] == "SHORT")).sum())
        fn = int(((valid["y_predict"] == "SHORT") & (valid["y_true"] == "LONG")).sum())
        pos = tp + fn
        neg = tn + fp
        tpr = float(tp / pos) if pos else 0.0
        tnr = float(tn / neg) if neg else 0.0
        accuracy = float((tp + tn) / n_valid) if n_valid else 0.0
        balance_gap = abs(tpr - tnr)
        score = accuracy - balance_penalty_weight * balance_gap
        accuracy_lcb = _wilson_lcb(accuracy=accuracy, n=n_valid)

        result_row = {
            "score": score,
            "accuracy": accuracy,
            "accuracy_lcb": accuracy_lcb,
            "tpr": tpr,
            "tnr": tnr,
            "balance_gap": balance_gap,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "n_valid": int(n_valid),
            "hyperparameters": {
                "authors": list(selected_authors),
                "window_to_analysis": int(window),
                "decay_rate": float(decay_rate),
                "decay_start_day": int(decay_start_day),
                "initial_weight": float(initial_weight),
            },
        }
        results.append(result_row)
        evaluated_cache[signature] = result_row

        trial.set_user_attr("accuracy", accuracy)
        trial.set_user_attr("accuracy_lcb", accuracy_lcb)
        trial.set_user_attr("balance_gap", balance_gap)
        trial.set_user_attr("n_valid", int(n_valid))
        trial.set_user_attr("authors", list(selected_authors))
        return float(score), float(n_valid)

    sampler = optuna.samplers.NSGAIISampler(seed=sampler_seed)
    study = optuna.create_study(
        directions=["maximize", "maximize"],
        sampler=sampler,
    )

    print("[optuna_tuner] Starting optimization...")
    for i in range(effective_trials):
        study.optimize(_objective, n_trials=1, catch=(Exception,))
        trial_no = i + 1
        trial = study.trials[-1]
        values = trial.values or [None, None]
        score_v = float(values[0]) if values[0] is not None else float("nan")
        n_valid_v = float(values[1]) if len(values) > 1 and values[1] is not None else float("nan")
        print(
            f"[optuna_tuner] trial {trial_no}/{effective_trials} "
            f"score={score_v:.3f} n_valid={n_valid_v:.0f} state={trial.state.name}"
        )
        if trial_no % max(1, int(report_every)) == 0:
            _report(trial_no=trial_no, done=False)

    top = _report(trial_no=effective_trials, done=True)
    print(f"[optuna_tuner] finished — saved top-{len(top)} → {save_path}")
    return top


BASE_DIR = Path(__file__).resolve().parent
TOP_SOURCES_PATH = BASE_DIR / "unbias_top_sources.json"
BASIC_PARAMS_PATH = BASE_DIR / "1_day_basic_parameters.json"
TUNING_OUTPUT_PATH = BASE_DIR / "tuning_top_optuna.json"
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

    today = datetime.utcnow().date()
    forecast_date = "2026-03-17"
    horizon = int(params.get("horizon", 1))
    n_trials = int(params.get("n_trials", 500))
    report_every = int(params.get("report_every", 20))

    print(
        f"[optuna_main] sources from unbias bucket={UNBIAS_BUCKET} "
        f"top_n={UNBIAS_TOP_N}: {len(authors)}"
    )
    print(
        f"[optuna_main] forecast_date={forecast_date} horizon={horizon} "
        f"n_trials={n_trials}"
    )

    from ...multiagent_graph import app

    top = find_best_hyperparameters_optuna(
        app=app,
        authors=authors,
        forecast_date=forecast_date,
        horizon=horizon,
        save_path=TUNING_OUTPUT_PATH,
        n_trials=n_trials,
        step=float(params.get("step", 0.1)),
        decay_rate_range=tuple(params.get("decay_rate_range", (0.05, 0.35, 0.05))),
        decay_start_day_range=tuple(params.get("decay_start_day_range", (3, 14, 1))),
        initial_weight_range=tuple(params.get("initial_weight_range", (0.5, 1.5, 0.1))),
        window_to_analysis_range=tuple(params.get("window_to_analysis_range", (7, 28, 1))),
        eval_days=int(params.get("eval_days", 100)),
        top_k=int(params.get("top_k", 50)),
        balance_penalty_weight=float(params.get("balance_penalty_weight", 1.0)),
        min_accuracy=float(params.get("min_accuracy", 0.57)),
        max_gap=float(params.get("max_gap", 0.3)),
        min_n_valid=int(params.get("min_n_valid", 70)),
        report_every=report_every,
        sampler_seed=int(params.get("sampler_seed", 42)),
    )
    print(f"[optuna_main] tuning finished. Saved: {TUNING_OUTPUT_PATH}")
    return top


if __name__ == "__main__":
    log_path = BASE_DIR / "logs_optuna.log"
    sys.stdout = open(str(log_path), "w", encoding="utf-8", buffering=1)
    sys.stderr = sys.stdout
    main()

