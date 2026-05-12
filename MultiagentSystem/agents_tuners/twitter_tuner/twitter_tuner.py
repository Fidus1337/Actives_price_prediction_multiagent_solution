import copy
import itertools
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from ...multiagent_predictions_module import add_y_true, make_one_prediction
from Logs.LoggingSystem.LoggingSystem import LoggingSystem


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


def _all_author_subsets(authors: list[str]) -> list[list[str]]:
    uniq = list(dict.fromkeys(authors))
    if not uniq:
        raise ValueError("authors pool must be non-empty")
    subsets: list[list[str]] = []
    for r in range(1, len(uniq) + 1):
        for combo in itertools.combinations(uniq, r):
            subsets.append(list(combo))
    return subsets


def find_best_hyperparameters(
    app,
    authors: list[str],
    forecast_date: str | None = None,
    horizon: int | None = None,
    save_path: str | Path | None = None,
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
) -> list[dict]:
    """Grid-search twitter-agent hyperparameters and persist the top-`top_k` by a
    balance-aware score.

    Each `*_range` is either `(low, high)` or `(low, high, step)`. When step is
    omitted the function-level `step` is used as a fallback.

    Sweeps every combination of author subsets × numeric hyperparameters, runs
    `eval_days` forecasts per trial against the shared base dataset (fetched once),
    then ranks trials by:
        score = accuracy - balance_penalty_weight * |TPR - TNR|
    where TPR = TP / (TP + FN) on LONG and TNR = TN / (TN + FP) on SHORT. This
    rewards both raw accuracy and an evenly-filled confusion-matrix diagonal
    (TP ≈ TN in rate terms).
    """
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

    decay_rate_grid = _build_float_grid(decay_rate_range, default_step=step)
    initial_weight_grid = _build_float_grid(initial_weight_range, default_step=step)
    decay_start_day_grid = _build_int_grid(decay_start_day_range, default_step=step)
    window_grid = _build_int_grid(window_to_analysis_range, default_step=step)
    author_subsets = _all_author_subsets(authors)

    total_trials = (
        len(decay_rate_grid)
        * len(initial_weight_grid)
        * len(decay_start_day_grid)
        * len(window_grid)
        * len(author_subsets)
    )
    print(
        f"[tuner] horizon={horizon}d  eval_days={eval_days}  step={step}  "
        f"total_trials={total_trials}  "
        f"(dr={len(decay_rate_grid)} iw={len(initial_weight_grid)} "
        f"dsd={len(decay_start_day_grid)} win={len(window_grid)} subs={len(author_subsets)})"
    )
    if total_trials == 0:
        raise ValueError("Empty grid — check ranges and step")

    load_dotenv(Path(__file__).resolve().parent.parent.parent.parent / "dev.env")
    cached_dataset = None
    print("[tuner] Twitter-only mode — skipping CoinGlass dataset fetch")

    forecast_dates = [
        (end_date - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(eval_days)
    ]

    if save_path is None:
        save_path = Path(__file__).resolve().parent / "tuning_top.json"
    else:
        save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    def _ranked_top(current_results: list[dict]) -> list[dict]:
        eligible = [
            r for r in current_results
            if r["accuracy"] >= min_accuracy and r["balance_gap"] <= max_gap
        ]
        ranked = sorted(
            eligible,
            key=lambda r: (r["score"], r["accuracy"], r["n_valid"]),
            reverse=True,
        )[:top_k]
        return [{"rank": i + 1, **r} for i, r in enumerate(ranked)]

    def _write_payload(top_list: list[dict], trial_no: int, done: bool) -> None:
        payload = {
            "horizon": horizon,
            "eval_days": eval_days,
            "step": step,
            "balance_penalty_weight": balance_penalty_weight,
            "min_accuracy": min_accuracy,
            "max_gap": max_gap,
            "total_trials": total_trials,
            "trials_completed": trial_no,
            "done": done,
            "top": top_list,
        }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def _report(current_results: list[dict], trial_no: int, done: bool) -> list[dict]:
        top_list = _ranked_top(current_results)
        print(
            f"\n[tuner] === running top-{len(top_list)} after {trial_no}/{total_trials} trials ==="
        )
        for r in top_list:
            hp = r["hyperparameters"]
            print(
                f"  #{r['rank']:<3} score={r['score']:.3f}  acc={r['accuracy']:.3f}  "
                f"tpr={r['tpr']:.3f}  tnr={r['tnr']:.3f}  gap={r['balance_gap']:.3f}  "
                f"n={r['n_valid']}  "
                f"win={hp['window_to_analysis']} dr={hp['decay_rate']} "
                f"dsd={hp['decay_start_day']} iw={hp['initial_weight']} "
                f"authors={hp['authors']}"
            )
        _write_payload(top_list, trial_no, done)
        print(f"[tuner] === end running top (saved → {save_path}) ===\n")
        return top_list

    results: list[dict] = []
    trial_idx = 0
    report_every = 100
    for subset in author_subsets:
        for window in window_grid:
            for decay_rate in decay_rate_grid:
                for decay_start_day in decay_start_day_grid:
                    for initial_weight in initial_weight_grid:
                        trial_idx += 1
                        trial_config = copy.deepcopy(base_config)
                        trial_config["forecast_start_date"] = forecast_date
                        trial_config["horizon"] = horizon
                        trial_config["agent_envolved_in_prediction"] = [
                            "agent_for_twitter_analysis"
                        ]
                        trial_config["agent_settings"]["agent_for_twitter_analysis"] = {
                            "window_to_analysis": int(window),
                            "decay_rate": float(decay_rate),
                            "decay_start_day": int(decay_start_day),
                            "initial_weight": float(initial_weight),
                            "authors": list(subset),
                        }

                        print(
                            f"\n[tuner] trial {trial_idx}/{total_trials} "
                            f"authors={subset} win={window} dr={decay_rate} "
                            f"dsd={decay_start_day} iw={initial_weight}"
                        )

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

                        results.append(
                            {
                                "score": score,
                                "accuracy": accuracy,
                                "tpr": tpr,
                                "tnr": tnr,
                                "balance_gap": balance_gap,
                                "tp": tp,
                                "tn": tn,
                                "fp": fp,
                                "fn": fn,
                                "n_valid": int(n_valid),
                                "hyperparameters": {
                                    "authors": list(subset),
                                    "window_to_analysis": int(window),
                                    "decay_rate": float(decay_rate),
                                    "decay_start_day": int(decay_start_day),
                                    "initial_weight": float(initial_weight),
                                },
                            }
                        )
                        print(
                            f"[tuner] → score={score:.3f}  acc={accuracy:.3f}  "
                            f"tpr={tpr:.3f}  tnr={tnr:.3f}  gap={balance_gap:.3f}  "
                            f"n_valid={n_valid}"
                        )

                        if trial_idx % report_every == 0:
                            _report(results, trial_idx, done=False)

    top = _report(results, trial_idx, done=True)
    print(f"[tuner] finished — saved top-{len(top)} → {save_path}")

    return top


if __name__ == "__main__":
    log_path = Path(__file__).resolve().parent / "logs.log"
    # sys.stdout = LoggingSystem(str(log_path), mode="w")
    
    sys.stdout = open(str(log_path), "w", encoding="utf-8", buffering=1)
    sys.stderr = sys.stdout


    from ...multiagent_system_main import app

    # Top authors by BULL/BEAR classified tweet count in twitter_archive.db
    # (neutral NO_CORRELATION_TO_BTC not counted; only 13 authors qualify)
    authors_pool = [
        "lookonchain",
        "rektcapital",
        "scottmelker",
        "ericbalchunas",
        "caprioleio",
        "ki_young_ju"
    ]

    find_best_hyperparameters(
        app=app,
        authors=authors_pool,
        forecast_date="2026-04-15",
        horizon=1,
        step=0.1,
        decay_rate_range=(0.20, 0.20, 1),
        decay_start_day_range=(3, 3, 3),
        initial_weight_range=(1, 1, 1),
        window_to_analysis_range=(7, 14, 7),
        eval_days=100,
        top_k=50,
    )
