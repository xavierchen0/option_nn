import polars as pl
import pandas as pd

PANDAS_SCHEMA = {
    # ID #
    "secid": pd.CategoricalDtype(),
    "cusip": pd.CategoricalDtype(),
    "sic": pd.CategoricalDtype(),
    "symbol": pd.CategoricalDtype(),
    "symbol_flag": pd.BooleanDtype(),
    "optionid": pd.Int32Dtype(),
    "ticker": pd.CategoricalDtype(),
    "exchange_d": pd.CategoricalDtype(),
    "issuer": pd.CategoricalDtype(),
    "issue_type": pd.CategoricalDtype(["0", "A", "7", "F", "%", "S", "U"]),
    "index_flag": pd.BooleanDtype(),
    "industry_group": pd.CategoricalDtype(),
    "class": pd.CategoricalDtype(),
    "root": pd.CategoricalDtype(),
    "suffix": pd.CategoricalDtype(),
  
    # OPTION TYPE #
    "cp_flag": pd.CategoricalDtype(["C", "P"]),
    "exercise_style": pd.CategoricalDtype(["A", "E", "?"]),
    "div_convention": pd.CategoricalDtype(["I", "F", "?"]),
    "ss_flag": pd.CategoricalDtype(["0", "1", "E"]),
    "am_settlement": pd.BooleanDtype(),
    "am_set_flag": pd.BooleanDtype(),
    "contract_size": pd.Int32Dtype(),
    "expiry_indicator": pd.CategoricalDtype(["w", "d", "m"]),

    # OPTION DATA #
    "strike_price": pd.Float32Dtype(),
    "best_bid": pd.Float32Dtype(),
    "best_offer": pd.Float32Dtype(),
    "impl_volatility": pd.Float32Dtype(),
    "volume": pd.Int32Dtype(),
    "open_interest": pd.Int32Dtype(),
    "forward_price": pd.Float32Dtype(),
    "cfadj": pd.Int32Dtype(),

    # ADDITIONAL OPTION DATA #
    "delta": pd.Float32Dtype(),
    "gamma": pd.Float32Dtype(),
    "vega": pd.Float32Dtype(),
    "theta": pd.Float32Dtype(),

    # DATETIME #
    "date": ("datetime", "%Y-%m-%d"),
    "Date": ("datetime", "%Y-%m-%d"),
    "last_date": ("datetime", "%Y-%m-%d"),
    "exdate": ("datetime", "%Y-%m-%d"),
    "expiration": ("datetime", "%Y-%m-%d"),

    # FORWARDS #
    "AMSettlement": pd.BooleanDtype(),
    "ForwardPrice": pd.Float32Dtype(),

    # INTEREST RATES #
    "days": pd.Int32Dtype(),
    "rate": pd.Float32Dtype(),

    # VIX #
    "vix": pd.Float32Dtype(),
    "vixh": pd.Float32Dtype(),
    "vixl": pd.Float32Dtype(),
    "vixo": pd.Float32Dtype(),
    "vxd": pd.Float32Dtype(),
    "vxdh": pd.Float32Dtype(),
    "vxdl": pd.Float32Dtype(),
    "vxdo": pd.Float32Dtype(),
    "vxn": pd.Float32Dtype(),
    "vxnh": pd.Float32Dtype(),
    "vxnl": pd.Float32Dtype(),
    "vxno": pd.Float32Dtype(),
    "vxo": pd.Float32Dtype(),
    "vxoh": pd.Float32Dtype(),
    "vxol": pd.Float32Dtype(),
    "vxoo": pd.Float32Dtype(),

    # NEW COLUMNS #
    "mid_price": pd.Float32Dtype(),
    "days_to_expiry": pd.Int32Dtype(),
    "moneyness": pd.Float32Dtype(),
    "log_moneyness": pd.Float32Dtype(),
}

def read_csv_with_schema(filepath, schema=PANDAS_SCHEMA, **kwargs):

    dtypes = {}
    date_formats = {}

    # Split dtypes into non-datetime and datetime, which are handled separately
    for col, dtype in schema.items():
        if isinstance(dtype, tuple) and dtype[0] == "datetime":
            dtypes[col] = pd.StringDtype()
            date_formats[col] = dtype[1]
        else:
            dtypes[col] = dtype

    # Read the file using the separated parameters
    df = pd.read_csv(filepath, dtype=dtypes, **kwargs)
    for date_col, date_format in date_formats.items():
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], format=date_format)

    return df


def enforce_dtypes(df, schema=PANDAS_SCHEMA):

    dtypes_to_correct = {}
    dt_cols_to_correct = {}

    for col in df.columns:
        expected_dtype = schema.get(col)

        # Checks for different than expected dtypes
        # Additional check for categories in categoricals if specified
        if (isinstance(expected_dtype, pd.CategoricalDtype) and (expected_dtype.categories is not None)):
            if (df.dtypes[col] != expected_dtype):
                dtypes_to_correct[col] = expected_dtype
        
        # Separate datetime logic
        elif (isinstance(expected_dtype, tuple) and (expected_dtype[0] == "datetime")):
            if (not pd.api.types.is_datetime64_any_dtype(df.dtypes[col])):
                dt_cols_to_correct[col] = expected_dtype[1]
        
        # Catch all else by type
        else:
            if not isinstance(df.dtypes[col], type(expected_dtype)):
                dtypes_to_correct[col] = expected_dtype

    df = df.astype(dtypes_to_correct)
    for date_col, date_format in dt_cols_to_correct.items():
        df[date_col] = pd.to_datetime(df[date_col], format=date_format)

    return df