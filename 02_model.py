import marimo

app = marimo.App()


with app.setup():
    from pathlib import Path

    import marimo as mo
    import polars as pl
    import torch
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, TensorDataset

    DATA_DIR = Path("data")
    DATA_DIR.mkdir(exist_ok=True)

    DEVICE = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"Using {DEVICE} device")


@app.cell(hide_code=True)
def prepare_data_md():
    mo.md(r"""
    # Prepare data

    Steps:
    1. Read input dataset via polars
    2. (paper) Train validation test split chronologically: 80% train, 20% test
    3. (paper) Train validation test split chronologically: 75% of train = train, 25% of train = validation
    4. Convert to torch
    """)


@app.cell
def prepare_data():
    # 1. Read input dataset via polars
    df = pl.read_parquet("data/cleaned_data.parquet")

    # 2. (paper) Train validation test split chronologically: 80% train, 20% test
    feature_cols = [
        "moneyness",
        "rate",
        "prev_vix",
        "years_to_expiry",
        "scaled_black_price",
    ]
    y_col = "scaled_market_price"
    k_col = "strike_price"

    X = df.select(feature_cols)

    y = df.select(y_col)

    K = df.select(k_col)

    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    split_idx = int(len(K) * 0.8)
    K_tmp, K_test = K.head(split_idx), K.tail(len(K) - split_idx)

    # 3. (paper) Train validation test split chronologically: 75% of train = train, 25% of train = validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.25, shuffle=False
    )

    split_idx = int(len(K_tmp) * 0.75)
    K_train, K_val = K_tmp.head(split_idx), K_tmp.tail(len(K_tmp) - split_idx)

    # Logging
    ui_elems1 = [
        mo.md("**Input DF:**"),
        df,
        mo.md(f"**Feature cols**: {feature_cols}"),
        mo.md(f"**Target col**: {y_col}"),
        mo.md(f"**Strike col**: {k_col}"),
        mo.md(
            f"**X_tmp**: Size = {len(X_tmp):,}, % of X = {round((len(X_tmp) / len(X)) * 100)}"
        ),
        mo.md(
            f"**X_test**: Size = {len(X_test):,}, % of X = {round((len(X_test) / len(X)) * 100)}"
        ),
        mo.md(
            f"**X_train**: Size = {len(X_train):,}, % of X_tmp = {round((len(X_train) / len(X_tmp)) * 100)}"
        ),
        mo.md(
            f"**X_val**: Size = {len(X_val):,}, % of X_tmp = {round((len(X_val) / len(X_tmp)) * 100)}"
        ),
        mo.md(
            f"**K_tmp**: Size = {len(K_tmp):,}, % of K = {round((len(K_tmp) / len(K)) * 100)}"
        ),
        mo.md(
            f"**K_test**: Size = {len(K_test):,}, % of K = {round((len(K_test) / len(K)) * 100)}"
        ),
        mo.md(
            f"**K_train**: Size = {len(K_train):,}, % of K_tmp = {round((len(K_train) / len(K_tmp)) * 100)}"
        ),
        mo.md(
            f"**K_val**: Size = {len(K_val):,}, % of K_tmp = {round((len(K_val) / len(K_tmp)) * 100)}"
        ),
    ]

    # 4. Convert to torch
    del X_tmp
    del K_tmp

    X_train, X_val, X_test = (
        X_train.to_torch(return_type="tensor", dtype=pl.Float32),
        X_val.to_torch(return_type="tensor", dtype=pl.Float32),
        X_test.to_torch(return_type="tensor", dtype=pl.Float32),
    )
    y_train, y_val, y_test = (
        y_train.to_torch(return_type="tensor", dtype=pl.Float32),
        y_val.to_torch(return_type="tensor", dtype=pl.Float32),
        y_test.to_torch(return_type="tensor", dtype=pl.Float32),
    )
    K_train, K_val, K_test = (
        K_train.to_torch(return_type="tensor", dtype=pl.Float32),
        K_val.to_torch(return_type="tensor", dtype=pl.Float32),
        K_test.to_torch(return_type="tensor", dtype=pl.Float32),
    )

    mo.vstack(ui_elems1, align="stretch")

    return X_train, X_val, X_test, y_train, y_val, y_test, K_train, K_val, K_test


if __name__ == "__main__":
    app.run()
