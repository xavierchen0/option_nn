import polars as pl
import pandas as pd

POLARS_SCHEMA = {
    # ID #
    "secid": pl.Int32,
    "cusip": pl.Int32,
    "sic": pl.Int32,
    "symbol": pl.String,
    "symbol_flag": pl.Int8,
    "optionid": pl.Int32,
    "ticker": pl.String,
    "exchange_d": pl.Int32,
    "issuer": pl.String,
    "issue_type": pl.Enum(["0", "A", "7", "F", "%", "S", "U"]),
    "index_flag": pl.Int8,
    "industry_group": pl.Int32,
    "class": pl.String,
    "root": pl.String,
    "suffix": pl.String,
  
    # OPTION TYPE #
    "cp_flag": pl.Enum(["C", "P"]),
    "exercise_style": pl.Enum(["A", "E", "?"]),
    "div_convention": pl.Enum(["I", "F", "?"]),
    "ss_flag": pl.Enum(["0", "1", "E"]),
    "am_settlement": pl.Int8,
    "am_set_flag": pl.Int8,
    "contract_size": pl.Int32,
    "expiry_indicator": pl.Enum(["w", "d", "m"]),

    # OPTION DATA #
    "strike_price": pl.Float32,
    "best_bid": pl.Float32,
    "best_offer": pl.Float32,
    "impl_volatility": pl.Float32,
    "volume": pl.Int32,
    "open_interest": pl.Int32,
    "forward_price": pl.Float32,
    "cfadj": pl.Int32,

    # ADDITIONAL OPTION DATA #
    "delta": pl.Float32,
    "gamma": pl.Float32,
    "vega": pl.Float32,
    "theta": pl.Float32,

    # DATETIME #
    "date": pl.Date,
    "Date": pl.Date,
    "last_date": pl.Date,
    "exdate": pl.Date,
    "expiration": pl.Date,

    # FORWARDS #
    "AMSettlement": pl.Int8,
    "ForwardPrice": pl.Float32,

    # INTEREST RATES #
    "days": pl.Int32,
    "rate": pl.Float32,

    # VIX #
    "vix": pl.Float32,
    "vixh": pl.Float32,
    "vixl": pl.Float32,
    "vixo": pl.Float32,
    "vxd": pl.Float32,
    "vxdh": pl.Float32,
    "vxdl": pl.Float32,
    "vxdo": pl.Float32,
    "vxn": pl.Float32,
    "vxnh": pl.Float32,
    "vxnl": pl.Float32,
    "vxno": pl.Float32,
    "vxo": pl.Float32,
    "vxoh": pl.Float32,
    "vxol": pl.Float32,
    "vxoo": pl.Float32,

    # NEW COLUMNS #
    "mid_price": pl.Float32,
    "days_to_expiry": pl.Int32,
    "moneyness": pl.Float32,
    "log_moneyness": pl.Float32,
}

OPT_ANALYSIS_COLS = [
    "date",
    "last_date",
    "exdate",
    "cp_flag",
    "strike_price",
    "best_bid",
    "best_offer",
    "impl_volatility",
    "volume",
    "open_interest",
]

