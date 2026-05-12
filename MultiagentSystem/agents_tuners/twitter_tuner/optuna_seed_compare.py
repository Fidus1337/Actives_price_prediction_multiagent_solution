import json
from datetime import date
from pathlib import Path

from ...multiagent_graph import app
from .twitter_optuna_tuner import (
    find_best_hyperparameters_optuna,
    _load_basic_params,
    _load_unique_sources_from_unbias,
)


BASE_DIR = Path(__file__).resolve().parent
TOP_SOURCES_PATH = BASE_DIR / "unbias_top_sources.json"
BASIC_PARAMS_PATH = BASE_DIR / "1_day_basic_parameters.json"
OUT_PREFIX = BASE_DIR / "tuning_top_optuna_seed"
SUMMARY_PATH = BASE_DIR / "tuning_top_optuna_seed_comparison.json"
SEEDS = [42, 123, 777]


def _extract_author_sets(top_rows: list[dict]) -> set[frozenset[str]]:
    out: set[frozenset[str]] = set()
    for row in top_rows:
        authors = row.get("hyperparameters", {}).get("authors", [])
        if authors:
            out.add(frozenset(authors))
    return out


def _read_top_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("top", [])


def main() -> dict:
    if not TOP_SOURCES_PATH.exists():
        raise FileNotFoundError(f"Missing file: {TOP_SOURCES_PATH}")
    if not BASIC_PARAMS_PATH.exists():
        raise FileNotFoundError(f"Missing file: {BASIC_PARAMS_PATH}")

    params = _load_basic_params(BASIC_PARAMS_PATH)
    authors = _load_unique_sources_from_unbias(TOP_SOURCES_PATH)
    if not authors:
        raise ValueError("No authors found in unbias_top_sources.json")

    horizon = int(params.get("horizon", 1))
    forecast_date = date.today().strftime("%Y-%m-%d")
    n_trials = int(params.get("n_trials", 120))

    print(
        f"[seed_compare] authors={len(authors)} horizon={horizon} "
        f"forecast_date={forecast_date} n_trials={n_trials}"
    )

    run_paths: dict[int, str] = {}
    for seed in SEEDS:
        out_path = Path(f"{OUT_PREFIX}_{seed}.json")
        print(f"[seed_compare] running seed={seed} -> {out_path.name}")
        find_best_hyperparameters_optuna(
            app=app,
            authors=authors,
            forecast_date=forecast_date,
            horizon=horizon,
            save_path=out_path,
            n_trials=n_trials,
            step=float(params.get("step", 0.1)),
            decay_rate_range=tuple(params.get("decay_rate_range", (0.05, 0.35, 0.05))),
            decay_start_day_range=tuple(params.get("decay_start_day_range", (3, 14, 1))),
            initial_weight_range=tuple(params.get("initial_weight_range", (0.5, 1.5, 0.1))),
            window_to_analysis_range=tuple(params.get("window_to_analysis_range", (7, 28, 1))),
            eval_days=int(params.get("eval_days", 100)),
            top_k=int(params.get("top_k", 100)),
            balance_penalty_weight=float(params.get("balance_penalty_weight", 1.0)),
            min_accuracy=float(params.get("min_accuracy", 0.57)),
            max_gap=float(params.get("max_gap", 0.3)),
            report_every=int(params.get("report_every", 20)),
            sampler_seed=seed,
        )
        run_paths[seed] = str(out_path)

    top_rows_by_seed = {seed: _read_top_rows(Path(path)) for seed, path in run_paths.items()}
    sets_by_seed = {seed: _extract_author_sets(rows) for seed, rows in top_rows_by_seed.items()}

    common_all = set.intersection(*(sets_by_seed[s] for s in SEEDS)) if all(sets_by_seed[s] for s in SEEDS) else set()

    pairwise = {}
    for i in range(len(SEEDS)):
        for j in range(i + 1, len(SEEDS)):
            a, b = SEEDS[i], SEEDS[j]
            inter = sets_by_seed[a] & sets_by_seed[b]
            union = sets_by_seed[a] | sets_by_seed[b]
            pairwise[f"{a}_{b}"] = {
                "intersection_count": len(inter),
                "union_count": len(union),
                "jaccard": (len(inter) / len(union)) if union else 0.0,
            }

    summary = {
        "seeds": SEEDS,
        "n_trials_each": n_trials,
        "run_paths": run_paths,
        "top_sizes": {seed: len(top_rows_by_seed[seed]) for seed in SEEDS},
        "unique_author_set_sizes": {seed: len(sets_by_seed[seed]) for seed in SEEDS},
        "common_author_set_count_all_seeds": len(common_all),
        "common_author_sets_all_seeds": [sorted(list(s)) for s in sorted(common_all, key=lambda x: (len(x), sorted(x)))],
        "pairwise_overlap": pairwise,
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[seed_compare] done. Summary saved -> {SUMMARY_PATH}")
    return summary


if __name__ == "__main__":
    main()

