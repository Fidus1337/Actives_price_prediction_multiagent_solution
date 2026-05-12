import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

from Classic_ml_solutions_rework.Dataset_pipeline.get_features import FeaturesGetterRework

UNBIAS_URL = "https://unbias.fyi/api/v1/sentiment"
CONFIG_PATH = _PROJECT_ROOT / "configs" / "multiagent_config.json"


def get_authors_signals(author: str, days: int, key: str, asset: str = "BTC") -> dict:
    r = requests.get(
        UNBIAS_URL,
        params={"asset": asset, "days": days, "handle": author},
        headers={"X-API-Key": key},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def fetch_author_df(author: str, days: int, key: str) -> pd.DataFrame:
    """Возвращает DataFrame [date, author, confidence] для одного автора."""
    payload = get_authors_signals(author, days, key)
    data = payload.get("data", [])
    if not data:
        return pd.DataFrame(columns=["date", "author", "confidence"])

    df = pd.DataFrame(data)[["date", "sentiment_score"]].rename(
        columns={"sentiment_score": "confidence"}
    )
    df["date"] = pd.to_datetime(df["date"])
    df["author"] = author
    df = df.drop_duplicates(subset=["date", "author"], keep="last")
    return df[["date", "author", "confidence"]]


def fetch_all_authors(
    authors: list[str], days: int, key: str, sleep_sec: float = 0.2
) -> pd.DataFrame:
    """Long-формат: [date, author, confidence] по всем авторам."""
    frames: list[pd.DataFrame] = []
    for a in authors:
        try:
            df = fetch_author_df(a, days, key)
            print(f"[ok]  {a:>18}  rows={len(df)}")
            frames.append(df)
        except Exception as e:
            print(f"[err] {a:>18}  {e}")
        time.sleep(sleep_sec)
    if not frames:
        return pd.DataFrame(columns=["date", "author", "confidence"])
    return pd.concat(frames, ignore_index=True).sort_values(["date", "author"])


def aggregate_signals(df_long: pd.DataFrame, threshold: float = 50.0) -> pd.DataFrame:
    """Long → wide + средний confidence + дискретный y_pred.

    Колонки результата:
      - date
      - {author}__conf на каждого автора (pivot)
      - confidence_mean: среднее по не-NaN авторам
      - voters_count:    сколько авторов проголосовало
      - y_pred:          'LONG' (>thr), 'SHORT' (<thr), 'NEUTRAL' (==thr), NaN если voters_count=0
    """
    if df_long.empty:
        return pd.DataFrame(columns=pd.Index(
            ["date", "confidence_mean", "voters_count", "y_pred"]
        ))

    wide = (
        df_long.pivot_table(index="date", columns="author", values="confidence", aggfunc="last")
               .sort_index()
    )
    wide.columns = pd.Index([f"{c}__conf" for c in wide.columns])
    wide = wide.reset_index()

    author_cols = [c for c in wide.columns if c.endswith("__conf")]
    wide["confidence_mean"] = wide[author_cols].mean(axis=1, skipna=True)
    wide["voters_count"] = wide[author_cols].notna().sum(axis=1).astype(int)
    wide.loc[wide["voters_count"] == 0, "confidence_mean"] = pd.NA

    y_pred = pd.Series(pd.NA, index=wide.index, dtype="object")
    cm = wide["confidence_mean"]
    y_pred[cm.gt(threshold)] = "LONG"
    y_pred[cm.lt(threshold)] = "SHORT"
    y_pred[cm.eq(threshold)] = "NEUTRAL"
    wide["y_pred"] = y_pred

    return wide


def apply_confidence_decay(
    wide: pd.DataFrame,
    window_to_analysis: int,
    decay_rate: float,
    decay_start_day: int,
    initial_weight: float,
    threshold: float = 50.0,
) -> pd.DataFrame:
    """Оконное агрегирование confidence_mean с экспоненциальным затуханием.

    Для каждой строки (anchor_date) берёт дни из окна
    [anchor_date - (window_to_analysis - 1), anchor_date] и усредняет
    confidence_mean с весами по возрасту:

        age < decay_start_day:  weight = 1.0
        age >= decay_start_day: weight = initial_weight * (1 - decay_rate) ** (age - decay_start_day)

    NaN-дни в окне пропускаются. Если не осталось ни одного дня — decayed = NaN.
    y_pred пересчитывается относительно confidence_mean_decayed.
    """
    if wide.empty or "confidence_mean" not in wide.columns:
        out = wide.copy()
        out["confidence_mean_decayed"] = pd.Series(dtype="float64")
        out["weights_sum"] = pd.Series(dtype="float64")
        out["days_in_window"] = pd.Series(dtype="int64")
        return out

    out = wide.sort_values("date").reset_index(drop=True).copy()
    dates = pd.to_datetime(out["date"]).dt.normalize()
    cm = out["confidence_mean"].astype("float64")

    decayed = [float("nan")] * len(out)
    wsum    = [float("nan")] * len(out)
    dcount  = [0] * len(out)

    for i in range(len(out)):
        anchor = dates.iloc[i]
        lo = anchor - pd.Timedelta(days=window_to_analysis - 1)
        j = i
        num = 0.0
        den = 0.0
        n_used = 0
        while j >= 0 and dates.iloc[j] >= lo:
            val = cm.iloc[j]
            if pd.notna(val):
                age = (anchor - dates.iloc[j]).days
                if age < decay_start_day:
                    w = 1.0
                else:
                    w = initial_weight * (1.0 - decay_rate) ** (age - decay_start_day)
                num += val * w
                den += w
                n_used += 1
            j -= 1
        if den > 0:
            decayed[i] = num / den
            wsum[i]    = den
            dcount[i]  = n_used

    out["confidence_mean_decayed"] = decayed
    out["weights_sum"] = wsum
    out["days_in_window"] = dcount

    y_pred = pd.Series(pd.NA, index=out.index, dtype="object")
    cmd = out["confidence_mean_decayed"]
    y_pred[cmd.gt(threshold)] = "LONG"
    y_pred[cmd.lt(threshold)] = "SHORT"
    y_pred[cmd.eq(threshold)] = "NEUTRAL"
    out["y_pred"] = y_pred

    return out


def add_y_true(df: pd.DataFrame, horizon: int, price_col: str = "close") -> pd.DataFrame:
    """Добавляет колонку y_true: LONG если price[t+h] > price[t], SHORT если меньше."""
    out = df.sort_values("date").reset_index(drop=True).copy()
    future = out[price_col].shift(-horizon)
    out[f"{price_col}_future_{horizon}d"] = future
    y = pd.Series(pd.NA, index=out.index, dtype="object")
    y[future > out[price_col]] = "LONG"
    y[future < out[price_col]] = "SHORT"
    out["y_true"] = y
    return out


def build_confusion_matrix(
    df: pd.DataFrame,
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
) -> tuple[pd.DataFrame, dict]:
    """Confusion matrix по готовым колонкам y_true / y_pred.

    NEUTRAL и NaN исключаются (сравниваем только LONG vs SHORT).
    Возвращает (cm, metrics).
    """
    labels = ["LONG", "SHORT"]
    mask = df[y_true_col].isin(labels) & df[y_pred_col].isin(labels)
    sub = df.loc[mask]

    cm = pd.crosstab(
        sub[y_true_col].astype(str),
        sub[y_pred_col].astype(str),
        rownames=["y_true"],
        colnames=["y_pred"],
        dropna=False,
    ).reindex(index=labels, columns=labels, fill_value=0)

    tp = int(cm.loc["LONG", "LONG"])
    tn = int(cm.loc["SHORT", "SHORT"])
    fp = int(cm.loc["SHORT", "LONG"])
    fn = int(cm.loc["LONG", "SHORT"])
    total = tp + tn + fp + fn

    metrics = {
        "n": total,
        "n_neutral_or_nan": int((~mask).sum()),
        "accuracy":        (tp + tn) / total if total else float("nan"),
        "precision_long":  tp / (tp + fp) if (tp + fp) else float("nan"),
        "recall_long":     tp / (tp + fn) if (tp + fn) else float("nan"),
        "precision_short": tn / (tn + fn) if (tn + fn) else float("nan"),
        "recall_short":    tn / (tn + fp) if (tn + fp) else float("nan"),
    }
    return cm, metrics


def last_dates_per_author(df_long: pd.DataFrame) -> pd.DataFrame:
    """Для каждого автора возвращает дату последнего сигнала и количество строк."""
    if df_long.empty:
        return pd.DataFrame(columns=pd.Index(["author", "last_date", "rows"]))
    grouped = df_long.groupby("author")["date"]
    agg = pd.DataFrame({
        "last_date": grouped.max(),
        "rows": grouped.count(),
    }).reset_index()
    return agg.sort_values("last_date", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    load_dotenv("dev.env")
    key = os.getenv("UNBIAS_KEY")
    if not key:
        raise RuntimeError("UNBIAS_KEY is not set in dev.env")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    twitter_cfg = cfg["agent_settings"]["agent_for_twitter_analysis"]
    authors: list[str] = ["CarpeNoctom",
                "JSeyff",
                "AltcoinPsycho",
                "DavidDuong",
                "TraderMercury",
                "_Checkmatey_",
                "CryptoHayes",
                "rektcapital"]
    DAYS = 200
    HORIZON = int(cfg.get("horizon", 1))

    WINDOW_TO_ANALYSIS = 14
    DECAY_RATE         = 0.10
    DECAY_START_DAY    = 1
    INITIAL_WEIGHT     = 1.0
    THRESHOLD          = 50.0

    print(f"fetching {len(authors)} authors, days={DAYS}")
    df_long = fetch_all_authors(authors, days=DAYS, key=key)

    out_dir = Path(__file__).resolve().parent
    long_csv = out_dir / "unbias_signals_long.csv"
    df_long.to_csv(long_csv, index=False)
    print(f"\nsaved long  -> {long_csv}  shape={df_long.shape}")

    last_dates = last_dates_per_author(df_long)
    print("\n=== last fetched tweet date per author ===")
    print(last_dates.to_string(index=False))

    if not df_long.empty:
        print(f"\noverall max date: {df_long['date'].max().date()}")
        print(f"overall min date: {df_long['date'].min().date()}")

    df_wide = aggregate_signals(df_long)
    df_wide = apply_confidence_decay(
        df_wide,
        window_to_analysis=WINDOW_TO_ANALYSIS,
        decay_rate=DECAY_RATE,
        decay_start_day=DECAY_START_DAY,
        initial_weight=INITIAL_WEIGHT,
        threshold=THRESHOLD,
    )
    wide_csv = out_dir / "unbias_signals_wide.csv"
    df_wide.to_csv(wide_csv, index=False)
    print(f"\nsaved wide  -> {wide_csv}  shape={df_wide.shape}")
    print(
        f"decay params: window={WINDOW_TO_ANALYSIS} rate={DECAY_RATE} "
        f"start_day={DECAY_START_DAY} init={INITIAL_WEIGHT} thr={THRESHOLD}"
    )

    if not df_wide.empty:
        print("\ny_pred distribution:")
        print(df_wide["y_pred"].value_counts(dropna=False).to_string())
        print("\nlast 10 rows:")
        print(df_wide[[
            "date", "confidence_mean", "confidence_mean_decayed",
            "weights_sum", "days_in_window", "voters_count", "y_pred",
        ]].tail(10).to_string(index=False))

        getter = FeaturesGetterRework(env_path=_PROJECT_ROOT / "dev.env")
        ohlcv = getter.get_feature(
            "spot_price_history",
            symbol="BTCUSDT",
            interval="1d",
            limit=1000,
        ).rename(columns={
            "spot_price_history__open":       "open",
            "spot_price_history__high":       "high",
            "spot_price_history__low":        "low",
            "spot_price_history__close":      "close",
            "spot_price_history__volume_usd": "volume",
        })

        dataset = df_wide.merge(ohlcv, on="date", how="inner")
        dataset = add_y_true(dataset, horizon=HORIZON)

        out_csv = out_dir / f"unbias_dataset_h{HORIZON}.csv"
        dataset.to_csv(out_csv, index=False)
        print(f"\nsaved merged-> {out_csv}  shape={dataset.shape}")

        cm, metrics = build_confusion_matrix(dataset)
        print(f"\n=== confusion matrix (horizon={HORIZON}d, n={metrics['n']}, skipped neutral/nan={metrics['n_neutral_or_nan']}) ===")
        print(cm)
        print(
            "\naccuracy={accuracy:.4f} | "
            "precision[LONG]={precision_long:.4f} recall[LONG]={recall_long:.4f} | "
            "precision[SHORT]={precision_short:.4f} recall[SHORT]={recall_short:.4f}".format(**metrics)
        )
