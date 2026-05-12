from concurrent.futures import ThreadPoolExecutor, as_completed

from MultiagentSystem.dataset_pipeline.FeaturesGetterModule.FeaturesGetter import FeaturesGetter


def get_features(getter: FeaturesGetter, API_KEY: str):
    tasks = [
        ("open_interest_history",
         lambda: getter.get_history(
             endpoint_name="open_interest_history",
             exchange="Bybit", symbol="BTCUSDT", interval="1d",
             prefix="futures_open_interest_history")),

        ("open_interest_aggregated",
         lambda: getter.get_history(
             endpoint_name="open_interest_aggregated",
             symbol="BTC", interval="1d",
             prefix="futures_open_interest_aggregated_history")),

        ("open_interest_stablecoin",
         lambda: getter.get_history(
             endpoint_name="open_interest_stablecoin",
             exchange_list="Bybit", symbol="BTC", interval="1d",
             prefix="futures_open_interest_aggregated_stablecoin_history")),

        ("open_interest_coin_margin",
         lambda: getter.get_history(
             endpoint_name="open_interest_coin_margin",
             exchange_list="Bybit", symbol="BTC", interval="1d",
             prefix="futures_open_interest_aggregated_coin_margin_history")),

        ("funding_rate_history",
         lambda: getter.get_history(
             endpoint_name="funding_rate_history",
             exchange="Bybit", symbol="BTCUSDT", interval="1d",
             prefix="futures_funding_rate_history")),

        ("funding_rate_oi_weight",
         lambda: getter.get_history(
             endpoint_name="funding_rate_oi_weight",
             symbol="BTC", interval="1d",
             prefix="futures_funding_rate_oi_weight_history")),

        ("funding_rate_vol_weight",
         lambda: getter.get_history(
             endpoint_name="funding_rate_vol_weight",
             symbol="BTC", interval="1d",
             prefix="futures_funding_rate_vol_weight_history")),

        ("global_long_short_account_ratio",
         lambda: getter.get_history(
             endpoint_name="global_long_short_account_ratio",
             exchange="Bybit", symbol="BTCUSDT", interval="1d",
             prefix="futures_global_long_short_account_ratio_history")),

        ("top_long_short_account_ratio",
         lambda: getter.get_history(
             endpoint_name="top_long_short_account_ratio",
             exchange="Bybit", symbol="BTCUSDT", interval="1d",
             prefix="futures_top_long_short_account_ratio_history")),

        ("top_long_short_position_ratio",
         lambda: getter.get_history(
             endpoint_name="top_long_short_position_ratio",
             exchange="Bybit", symbol="BTCUSDT", interval="1d",
             prefix="futures_top_long_short_position_ratio_history")),

        ("net_position",
         lambda: getter.get_history(
             endpoint_name="net_position",
             exchange="Bybit", symbol="BTCUSDT", interval="1d",
             prefix="futures_v2_net_position_history")),

        ("liquidation_history",
         lambda: getter.get_history(
             endpoint_name="liquidation_history",
             exchange="Bybit", symbol="BTCUSDT", interval="1d",
             prefix="futures_liquidation_history")),

        ("liquidation_aggregated",
         lambda: getter.get_history(
             endpoint_name="liquidation_aggregated",
             exchange_list="Bybit", symbol="BTC", interval="1d",
             prefix="futures_liquidation_aggregated_history")),

        ("orderbook_ask_bids",
         lambda: getter.get_history(
             endpoint_name="orderbook_ask_bids",
             exchange="Bybit", symbol="BTCUSDT", interval="1d",
             prefix="futures_orderbook_ask_bids_history")),

        ("orderbook_aggregated",
         lambda: getter.get_history(
             endpoint_name="orderbook_aggregated",
             exchange_list="Bybit", symbol="BTC", interval="1d",
             prefix="futures_orderbook_aggregated_ask_bids_history")),

        ("taker_buy_sell_volume",
         lambda: getter.get_history(
             endpoint_name="taker_buy_sell_volume",
             exchange="Bybit", symbol="BTCUSDT", interval="1d",
             prefix="futures_v2_taker_buy_sell_volume_history")),

        ("taker_buy_sell_volume_aggregated",
         lambda: getter.get_history(
             endpoint_name="taker_buy_sell_volume_aggregated",
             exchange_list="Bybit", symbol="BTC", interval="1d",
             prefix="futures_aggregated_taker_buy_sell_volume_history")),

        ("btc_lth_supply",
         lambda: getter.get_bitcoin_lth_supply(
             pct_window=30, z_window=180, slope_window=14,
             prefix="index_btc_lth_supply")),

        ("btc_active_addresses",
         lambda: getter.get_bitcoin_active_addresses(
             pct_window=7, z_window=180, slope_window=14,
             prefix="index_btc_active_addresses")),

        ("btc_sth_supply",
         lambda: getter.get_bitcoin_sth_supply(
             pct_window=30, z_window=180, slope_window=14,
             prefix="index_btc_sth_supply")),

        ("bitfinex_margin",
         lambda: _fetch_bitfinex_margin(getter)),

        ("cgdi_index",
         lambda: _fetch_cgdi_index(getter)),

        ("coinbase_premium",
         lambda: _fetch_coinbase_premium(getter)),

        ("btc_reserve_risk",
         lambda: getter.get_bitcoin_reserve_risk(
             z_window=180, slope_window=14,
             prefix="index_btc_reserve_risk")),

        ("spot_price_history",
         lambda: _fetch_spot(getter)),

        ("sp500",
         lambda: getter.get_sp500_ohlcv(days=1250, prefix="sp500")),

        ("gold",
         lambda: getter.get_gold_ohlcv(days=1250, prefix="gold")),

        ("igv",
         lambda: getter.get_igv_ohlcv(days=1250, prefix="igv")),
    ]

    results = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_idx = {
            executor.submit(fn): (i, name)
            for i, (name, fn) in enumerate(tasks)
        }
        for future in as_completed(future_to_idx):
            i, name = future_to_idx[future]
            results[i] = future.result()
            print(f"  [fetched] {name}")

    return [
        results[0],   # df_oi
        results[1],   # df_oi_agg
        results[2],   # df_stable_oi
        results[3],   # df_coin_margin
        results[4],   # df_funding
        results[5],   # df_oi_weight_funding
        results[6],   # df_vol_weight_funding
        results[7],   # df_ls_accounts
        results[8],   # df_top_ls_accounts
        results[9],   # df_top_ls_positions
        results[10],  # df_net_pos
        results[11],  # df_liq
        results[12],  # df_liq_agg
        results[13],  # df_ob
        results[14],  # df_ob_agg
        results[15],  # df_taker
        results[16],  # df_taker_agg
        results[17],  # bitfinex_margin_ls_df
        results[18],  # futures_cgdi_index_df
        results[19],  # coinbase_premium_df
        results[20],  # df_lth_supply
        results[21],  # df_aa
        results[22],  # df_sth_supply
        results[23],  # df_rr
        results[24],  # df_spot
        results[25],  # df_sp500
        results[26],  # df_gold
        results[27],  # df_igv
    ]


# ---------------------------------------------------------------------------
# Helpers для датасетов с post-processing (rename columns)
# ---------------------------------------------------------------------------

def _fetch_bitfinex_margin(getter: FeaturesGetter):
    df = getter.get_bitfinex_margin_long_short(symbol="BTC", interval="1d", prefix="bfx_margin")
    return df.rename(columns={
        "bfx_margin__long_quantity": "long_quantity",
        "bfx_margin__short_quantity": "short_quantity",
    })


def _fetch_cgdi_index(getter: FeaturesGetter):
    df = getter.get_cgdi_index(interval="1d", prefix="cgdi")
    return df.rename(columns={
        "cgdi__index_value":   "cgdi_index_value",
        "cgdi__log_level":     "cgdi_log_level",
        "cgdi__dev_from_base": "cgdi_dev_from_base",
        "cgdi__dev_softsign":  "cgdi_dev_softsign",
    })


def _fetch_coinbase_premium(getter: FeaturesGetter):
    df = getter.get_coinbase_premium_index(interval="1d", prefix="premium")
    return df.rename(columns={
        "premium__premium":          "premium",
        "premium__premium_rate":     "premium_rate",
        "premium__premium_abs":      "cb_premium_abs",
        "premium__premium_softsign": "cb_premium_softsign",
        "premium__premium_rate_bps": "cb_premium_rate_bps",
        "premium__implied_ref_price": "cb_implied_ref_price",
    })


def _fetch_spot(getter: FeaturesGetter):
    df = getter.get_history(
        endpoint_name="spot_price_history",
        exchange="Bybit", symbol="BTCUSDT", interval="1d",
        prefix="",
    )
    df.columns = df.columns.str.lstrip("_")
    return df
