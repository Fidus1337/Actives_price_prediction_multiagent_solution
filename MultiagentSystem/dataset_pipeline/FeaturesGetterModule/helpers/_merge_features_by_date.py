from functools import reduce
import pandas as pd


def _dedupe_by_date(df: pd.DataFrame, how: str = "last") -> pd.DataFrame:
    """
    Убирает дубли по date.
    how:
      - "last": оставляет последнюю строку по date
      - "first": оставляет первую
      - "mean": усредняет все numeric колонки внутри одного date (остальные игнор)
    """
    if df is None or df.empty:
        return df
    if "date" not in df.columns:
        raise ValueError("DataFrame must contain 'date' column")

    df = df.copy()
    df["date"] = df["date"].astype("string")

    if how in ("last", "first"):
        keep = how
        return df.sort_values("date", kind="stable").drop_duplicates("date", keep=keep).reset_index(drop=True)

    if how == "mean":
        num_cols = [c for c in df.columns if c != "date" and pd.api.types.is_numeric_dtype(df[c])]
        # если numeric нет — просто дедыуп по last
        if not num_cols:
            return df.sort_values("date", kind="stable").drop_duplicates("date", keep="last").reset_index(drop=True)
        out = df.groupby("date", as_index=False)[num_cols].mean()
        return out.sort_values("date", kind="stable").reset_index(drop=True)

    raise ValueError("how must be one of: 'last', 'first', 'mean'")


def merge_by_date(dfs: list[pd.DataFrame], how: str = "outer", dedupe: str = "last") -> pd.DataFrame:
    """
    Объединяет список DF по колонке 'date'.
    how: 'outer' (рекомендую) или 'inner'
    dedupe: см. _dedupe_by_date
    """
    # отфильтровать пустые
    cleaned = []
    for df in dfs:
        if df is None or df.empty:
            continue
        cleaned.append(_dedupe_by_date(df, how=dedupe))

    if not cleaned:
        return pd.DataFrame(columns=["date"])

    merged = reduce(lambda left, right: left.merge(right, on="date", how=how), cleaned)
    merged = merged.sort_values("date", kind="stable").reset_index(drop=True)
    return merged
