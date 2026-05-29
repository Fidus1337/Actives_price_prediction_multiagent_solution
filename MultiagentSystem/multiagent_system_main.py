import json
import os
import sys
from pathlib import Path

import pandas as pd

from .multiagent_predictions_module import (
    add_y_true,
    build_confusion_matrix,
    make_prediction_for_last_N_days,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / "dev.env")

from .multiagent_graph import app

from Logs.LoggingSystem.LoggingSystem import LoggingSystem


if __name__ == "__main__":
    log_path = Path(__file__).parent / "logs.log"
    sys.stdout = LoggingSystem(str(log_path), mode="w")

    # Load multiagent system config
    config_path = Path(__file__).resolve().parent.parent / "configs" / "multiagent_config.json"
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    # Last days to analyis
    N_days = 3

    load_dotenv(Path(__file__).resolve().parent.parent / "dev.env")
    os.environ["COINGLASS_API_KEY"]  # fail fast if key is missing

    cm_path = Path(__file__).parent / "confusion_matrix.png"
    save_path = Path(__file__).parent / "predictions_results.csv"

    # Gather predictions

    print(f"{N_days} | {cm_path}")
    make_prediction_for_last_N_days(
        app, config, N_days,
        checkpoint_every=10,
        cm_path=cm_path,
        save_results=True,
        save_path=str(save_path),
    )

    # Read saved per-row CSV and enrich it with y_true in a single batch.
    saved_df = pd.read_csv(save_path)
    saved_df = add_y_true(saved_df, config["horizon"])
    saved_df.to_csv(save_path, index=False)
    print(f"\n[OK] Predictions saved -> {save_path}")

    build_confusion_matrix(saved_df, config["horizon"], cm_path)
