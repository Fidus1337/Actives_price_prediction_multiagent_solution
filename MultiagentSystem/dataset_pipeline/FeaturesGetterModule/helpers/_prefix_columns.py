import pandas as pd


def _prefix_columns(df: pd.DataFrame, prefix: str, keep: tuple[str, ...] = ("date",)) -> pd.DataFrame:
    """
    Переименовывает все колонки, кроме указанных в keep, добавляя prefix + '__'.
    Пример: open -> futures_open_interest_history__open
    """
    rename_map = {c: f"{prefix}__{c}" for c in df.columns if c not in keep}
    return df.rename(columns=rename_map)


# ============================================================================
# Примеры использования
# ============================================================================
if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Пример 1: Базовое использование
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Пример 1: Базовое использование")
    print("=" * 60)
    
    df = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "open": [100, 110, 105],
        "close": [105, 108, 112],
        "volume": [1000, 1200, 900],
    })
    
    print("До:")
    print(df)
    print()
    
    df_prefixed = _prefix_columns(df, "oi")
    print("После _prefix_columns(df, 'oi'):")
    print(df_prefixed)
    print()
    
    # -------------------------------------------------------------------------
    # Пример 2: Указание нескольких колонок для сохранения
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Пример 2: Сохранение нескольких колонок без префикса")
    print("=" * 60)
    
    df2 = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02"],
        "symbol": ["BTC", "BTC"],
        "open": [50000, 51000],
        "close": [51000, 50500],
    })
    
    print("До:")
    print(df2)
    print()
    
    df2_prefixed = _prefix_columns(df2, "funding", keep=("date", "symbol"))
    print("После _prefix_columns(df2, 'funding', keep=('date', 'symbol')):")
    print(df2_prefixed)
    print()
    
    # -------------------------------------------------------------------------
    # Пример 3: Merge двух DataFrame с разными префиксами
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Пример 3: Merge двух DataFrame")
    print("=" * 60)
    
    df_oi = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02"],
        "open": [100, 110],
        "close": [105, 115],
    })
    
    df_funding = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02"],
        "open": [0.01, 0.02],
        "close": [0.015, 0.018],
    })
    
    print("DataFrame 1 (Open Interest):")
    print(df_oi)
    print()
    print("DataFrame 2 (Funding Rate):")
    print(df_funding)
    print()
    
    # Добавляем префиксы
    df_oi = _prefix_columns(df_oi, "oi")
    df_funding = _prefix_columns(df_funding, "funding")
    
    # Объединяем по date
    df_merged = df_oi.merge(df_funding, on="date")
    
    print("После merge с префиксами:")
    print(df_merged)
    print()
    
    print("=" * 60)
    print("Все примеры выполнены!")
    print("=" * 60)
