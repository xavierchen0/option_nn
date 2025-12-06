import marimo

__generated_with = "0.18.1"
app = marimo.App()

with app.setup:
    from pathlib import Path

    import lets_plot as lp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import polars as pl
    import QuantLib as ql
    import seaborn as sns
    from py_vollib.black import black

    # START_DATE = "2025-07-01"
    # END_DATE = "2025-08-31"
    START_DATE = "2020-09-01"
    END_DATE = "2025-08-29"

    DATA_DIR = Path("data")
    DATA_DIR.mkdir(exist_ok=True)


@app.function
def to_ql_date(d):
    return ql.Date(d.day, d.month, d.year)


@app.cell(hide_code=True)
def data_sources():
    mo.md(r"""
    # Data sources

    ## Query Data Range
    29 Aug 2020 - 29 Aug 2025

    ## Options
    Options data is queried from OptionMetrics through WRDS [link](https://wrds-www.wharton.upenn.edu/pages/get-data/optionmetrics/ivy-db-us/options/option-prices/)

    ## Forward Prices
    Forward Prices data is queried from OptionMetrics through WRDS [link](https://wrds-www.wharton.upenn.edu/pages/get-data/optionmetrics/ivy-db-us/options/forward-price/)

    ## Interest Rates
    Interest Rates data is queried from OptionMetrics through WRDS [link](https://wrds-www.wharton.upenn.edu/pages/get-data/optionmetrics/ivy-db-us/market/zero-coupon-yield-curve/)

    ## VIX
    VIX data is queried from CBOE Options through WRDS [link](https://wrds-www.wharton.upenn.edu/pages/get-data/cboe-indexes/cboe-indexes-1/cboe-indexes/)
    """)
    return


@app.cell(hide_code=True)
def read_md():
    mo.md(r"""
    # Read datasets
    """)
    return


@app.cell
def read_options():
    # options = pd.read_csv(DATA_DIR / "options.csv").sort_values(by="date")
    options = (
        pl.read_csv(DATA_DIR / "options.csv")
        .to_pandas(use_pyarrow_extension_array=True)
        .sort_values(by="date")
    )
    options
    return (options,)


@app.cell
def read_forwards():
    forwards = (
        pl.read_csv(DATA_DIR / "forwards.csv")
        .to_pandas(use_pyarrow_extension_array=True)
        .sort_values(by="date")
    )
    forwards
    return (forwards,)


@app.cell
def read_interests():
    interests = (
        pl.read_csv(DATA_DIR / "interest.csv")
        .to_pandas(use_pyarrow_extension_array=True)
        .sort_values(by="date")
    )
    interests
    return (interests,)


@app.cell
def read_vix():
    vix = (
        pl.read_csv(DATA_DIR / "vix.csv")
        .to_pandas(use_pyarrow_extension_array=True)
        .sort_values(by="Date")
    )
    vix
    return (vix,)


@app.cell(hide_code=True)
def clean_options_md():
    mo.md(r"""
    ## Clean Options

    Steps taken:
    1. Keep necessary columns
    2. Set the right dtypes
    3. Filter for specific dates
    4. Remove rows with null IVs due to special settlement type
    5. Divide strike by 1000; In OptionMetric, strike is x 1000
    6. (paper) Filter out no open interest and no volume
    7. Filter out best_bid == 0 as they are not representative
    8. Calculate mid option price
    9. (paper) Filter mid_price < 1/8
    10. Calculate days to expiry
        - 10a. -1 day for am settled options
    11. Calculate days to expiry in years
    12. (paper) Filter days_to_expiry > 120

    Notes:
    1. They are all European options
    """)
    return


@app.cell
def clean_options(options):
    # 1. Keep necessary columns
    options_tmp = options.loc[
        :,
        [
            "date",
            "cp_flag",
            "strike_price",
            "exdate",
            "impl_volatility",
            "best_bid",
            "best_offer",
            "am_settlement",
            "open_interest",
            "volume",
        ],
    ].copy()

    print("Option's columns: ", "\n", options_tmp.columns, "\n")

    # 2. Set the right dtypes
    options_tmp["date"] = pd.to_datetime(options_tmp["date"])
    options_tmp["exdate"] = pd.to_datetime(options_tmp["exdate"])

    options_tmp = options_tmp.astype(
        {
            "cp_flag": "category",
            "strike_price": "Int32",
            "impl_volatility": "Float32",
            "best_bid": "Float32",
            "best_offer": "Float32",
            "open_interest": "Int32",
            "volume": "Int32",
        }
    )

    print("Option's datatype check: ", "\n", options_tmp.dtypes, "\n")

    # 3. Filter for specific dates
    options_date_mask = (options_tmp["date"] >= START_DATE) & (
        options_tmp["date"] <= END_DATE
    )
    options_tmp = options_tmp[options_date_mask].copy()

    print("Maximum date: ", options_tmp["date"].max(), "\n")
    print("Minimum date: ", options_tmp["date"].min(), "\n")

    # 4. Remove rows with null IVs due to special settlement type
    print("DF size before dropping null IV: ", len(options_tmp), "\n")

    options_tmp = options_tmp.dropna(axis=0, subset=["impl_volatility"])

    print("DF size after dropping null IV: ", len(options_tmp), "\n")

    # 5. Divide strike by 1000; In OptionMetric, strike is x 1000
    options_tmp["strike_price"] = options_tmp["strike_price"] / 1000
    options_tmp = options_tmp.astype({"strike_price": "Int32"})

    # 6. (paper) filter out no open interest and no volume
    print(
        "DF size before filtering out no open interest and no volume: ",
        len(options_tmp),
        "\n",
    )

    no_open_interest_volume_mask = (options_tmp["open_interest"] == 0) | (
        options_tmp["volume"] == 0
    )

    print(
        "DF size for options with no open interest and no open volume: ",
        len(options_tmp[no_open_interest_volume_mask]),
        "\n",
    )

    options_tmp = (
        options_tmp.loc[~no_open_interest_volume_mask]
        .drop(columns=["open_interest", "volume"])
        .copy()
    )

    print(
        "DF size after filtering out no open interest and no volume: ",
        len(options_tmp),
        "\n",
    )

    # 7. filter out best_bid == 0 as they are not representative
    print(
        "DF size before filtering out best_bid == 0: ",
        len(options_tmp),
        "\n",
    )

    best_bid_0_mask = options_tmp["best_bid"] == 0

    print(
        "DF size for options with best_bid == 0: ",
        len(options_tmp[best_bid_0_mask]),
        "\n",
    )

    options_tmp = options_tmp.loc[~best_bid_0_mask].copy()

    print(
        "DF size after filtering out best_bid == 0: ",
        len(options_tmp),
        "\n",
    )

    # 8. calculate mid option price
    options_tmp["mid_price"] = (options_tmp["best_bid"] + options_tmp["best_offer"]) / 2
    options_tmp = options_tmp.drop(columns=["best_bid", "best_offer"])

    # 9. (paper) filter mid_price < 1/8
    print(
        "DF size before filtering out mid_price < 1/8: ",
        len(options_tmp),
        "\n",
    )

    mid_price_less_one_eighth_mask = options_tmp["mid_price"] < 0.125

    print(
        "DF size of options with mid_price < 1/8: ",
        len(options_tmp[mid_price_less_one_eighth_mask]),
        "\n",
    )

    options_tmp = options_tmp[~mid_price_less_one_eighth_mask].copy()

    print(
        "DF size after filtering out mid_price < 1/8: ",
        len(options_tmp),
        "\n",
    )

    # 10. Calculate days to expiry
    options_tmp["days_to_expiry"] = (
        options_tmp["exdate"] - options_tmp["date"]
    ).dt.days
    options_tmp = options_tmp.astype({"days_to_expiry": "Int32"})

    # 10a. -1 day for am settled options
    am_settled_mask = options_tmp["am_settlement"] == 1
    options_tmp.loc[am_settled_mask, "days_to_expiry"] = (
        options_tmp.loc[am_settled_mask, "days_to_expiry"] - 1
    )

    # 11. Calculate days to expiry in years
    options_tmp["years_to_expiry"] = options_tmp["days_to_expiry"] / 365
    options_tmp = options_tmp.astype({"years_to_expiry": "Float32"})

    # 12. (paper) Filter days_to_expiry > 120
    print(
        "DF size before filtering out days_to_expiry > 120: ",
        len(options_tmp),
        "\n",
    )

    days_to_expiry_more_120_mask = options_tmp["days_to_expiry"] > 120

    print(
        "DF size of options with days_to_expiry > 120: ",
        len(options_tmp[days_to_expiry_more_120_mask]),
        "\n",
    )

    options_tmp = options_tmp.loc[~days_to_expiry_more_120_mask].copy()

    print(
        "DF size after filtering out days_to_expiry > 120: ",
        len(options_tmp),
        "\n",
    )

    options_tmp
    return (options_tmp,)


@app.cell(hide_code=True)
def clean_forwards_md():
    mo.md(r"""
    ## Clean Forwards

    Steps taken:
    1. Keep necessary columns
    2. Set the right dtypes
    3. Filter for specific dates
    """)
    return


@app.cell
def clean_forwards(forwards):
    # 1. Keep necessary columns
    forwards_tmp = forwards.loc[
        :,
        ["date", "ForwardPrice", "expiration", "AMSettlement"],
    ].copy()

    print("Forward's columns: ", "\n", forwards_tmp.columns, "\n")

    # 2. Set the right dtypes
    forwards_tmp["date"] = pd.to_datetime(forwards_tmp["date"])
    forwards_tmp["expiration"] = pd.to_datetime(forwards_tmp["expiration"])

    forwards_tmp = forwards_tmp.astype(
        {
            "ForwardPrice": "Float32",
        }
    )

    print("Forward's datatype check: ", "\n", forwards_tmp.dtypes, "\n")

    # 3. Filter for specific dates
    forwards_date_mask = (forwards_tmp["date"] >= START_DATE) & (
        forwards_tmp["date"] <= END_DATE
    )
    forwards_tmp = forwards_tmp[forwards_date_mask].copy()

    print("Maximum date: ", forwards_tmp["date"].max(), "\n")
    print("Minimum date: ", forwards_tmp["date"].min(), "\n")

    forwards_tmp
    return (forwards_tmp,)


@app.cell(hide_code=True)
def clean_interest_md():
    mo.md(r"""
    ## Clean interest

    Steps taken:
    1. Set the right dtypes
    2. Find unique dates
    3. Create array of QuantLib Zero Curve objects
    4. For each unique date, fit the MonotonicCubicZeroCurve
    """)
    return


@app.cell
def clean_interest(interests):
    interests_tmp = interests.copy()

    # 1. Set the right dtypes
    interests_tmp["date"] = pd.to_datetime(interests_tmp["date"])

    interests_tmp = interests_tmp.astype({"days": "Int32", "rate": "Float64"})

    # 2. Find unique dates
    dates_idx = pd.Index(interests_tmp["date"].unique())

    # 3. Create array of QuantLib Zero Curve objects
    zc_series = pd.Series(
        index=interests_tmp["date"].unique(), name="ZeroCurve", dtype="object"
    )

    # 4. For each unique date, fit the MonotonicCubicZeroCurve
    for curve_date in dates_idx:
        specific_date_mask = interests_tmp["date"] == curve_date
        df_target = interests_tmp[specific_date_mask].copy()

        base_rate = df_target.sort_values("date")["rate"].iloc[0]
        base_row = pd.DataFrame([{"date": curve_date, "days": 0, "rate": base_rate}])
        df_target = pd.concat([base_row, df_target], axis="rows").sort_values("days")

        df_target["maturity"] = (
            df_target["date"] + df_target["days"] * pd.Timedelta(days=1)
        ).apply(to_ql_date)

        zc = ql.MonotonicCubicZeroCurve(
            df_target["maturity"].values,
            df_target["rate"],
            ql.Actual365Fixed(),
            ql.UnitedStates(ql.UnitedStates.GovernmentBond),
        )
        zc_series.loc[curve_date] = zc
    return


@app.cell(hide_code=True)
def clean_vix_md():
    mo.md(r"""
    ## Clean VIX

    Steps taken:
    1. Keep necessary columns
    2. Set the right dtypes
    3. Change from percentage to decimal
    4. (paper) Used the VIX closing level of the previous day as the standard deviation parameter
    5. Forward fill prev_vix for null values
    """)


@app.cell
def clean_vix(vix):
    # 1. Keep necessary columns
    vix_tmp = vix.loc[:, ["Date", "vix"]].copy()

    # 2. Set the right dtypes
    vix_tmp["Date"] = pd.to_datetime(vix_tmp["Date"])

    vix_tmp = vix_tmp.astype({"vix": "Float32"})

    print("Vix's datatype check: ", "\n", vix_tmp.dtypes, "\n")

    # 3. Change from percentage to decimal
    vix_tmp["vix"] = vix_tmp["vix"] / 100

    # 4. (paper) Used the VIX closing level of the previous day as the standard deviation parameter
    vix_tmp["prev_vix"] = vix_tmp["vix"].shift(1)
    vix_tmp = vix_tmp.astype({"prev_vix": "Float32"})

    # 5. Forward fill prev_vix for null values
    vix_tmp = vix_tmp.sort_values("Date")
    vix_tmp["prev_vix"] = vix_tmp["prev_vix"].ffill()

    vix_tmp

    return vix_tmp


@app.cell
def get_rate_function(ql, to_ql_date, zc_series):
    def get_rate(current_date, days_to_expiry):
        return (
            zc_series[current_date]
            .zeroRate(
                to_ql_date(current_date) + days_to_expiry,
                ql.Actual365Fixed(),
                ql.Continuous,
                ql.NoFrequency,
                True,
            )
            .rate()
        )

    return (get_rate,)


@app.cell
def plot_rate_curve(get_rate):
    def plot_curve(current_date, period=pd.DateOffset(years=5)):
        current_date = pd.to_datetime(current_date)

        date_range = pd.date_range(current_date, freq="D", end=current_date + period)
        df_plot = pd.DataFrame(
            np.arange(len(date_range)), index=date_range, columns=["days_to_expiry"]
        )
        df_plot["rate"] = df_plot.apply(
            lambda x: get_rate(current_date, x["days_to_expiry"]), axis="columns"
        )

        # Plot
        p = (
            lp.ggplot(df_plot, lp.aes(x="days_to_expiry", y="rate"))
            + lp.geom_line(color="#3f51b5", size=1.2)
            + lp.ggtitle(f"Interpolated Zero Curve: {current_date.date()}")
            + lp.labs(x="Days from Valuation", y="Zero Rate")
            + lp.theme_minimal()
        )

        return p

    # Render plot
    p = plot_curve("2021-01-29")
    p
    return


@app.cell(hide_code=True)
def merge_md():
    mo.md(r"""
    # Merge

    Steps taken:
    1. Compute risk-free rate
    2. Compute moneyness $\frac{F_t}{K}$
    3. Compute log moneyness $\log \frac{F_t}{K}$
    4. (paper) define ATM, OTM, ITM options
        - OTM: $\frac{F_t}{K} < 0.97$
        - ATM: $0.97 \leq \frac{F_t}{K} < 1.03$
        - ITM: $\frac{F_t}{K} \geq 1.03$
    5. Compute black's option price
    6. (paper) Scale the Black's option price
    7. (paper) Scale the Market option price
    """)
    return


@app.cell
def merge(forwards_tmp, options_tmp, vix_tmp, get_rate):
    combined = options_tmp.merge(
        forwards_tmp,
        left_on=["date", "exdate", "am_settlement"],
        right_on=["date", "expiration", "AMSettlement"],
        how="left",
    ).reset_index(drop=True)

    combined = combined.merge(
        vix_tmp, left_on="date", right_on="Date", how="left"
    ).reset_index(drop=True)

    combined = combined.drop(
        columns=["expiration", "am_settlement", "AMSettlement", "Date"]
    ).sort_values(by="date")

    # 1. Compute risk-free rate
    combined["rate"] = combined.apply(
        lambda x: get_rate(x["date"], x["days_to_expiry"]), axis="columns"
    )
    combined = combined.astype({"rate": "Float32"})

    # 2. Compute moneyness
    combined["moneyness"] = combined["ForwardPrice"] / combined["strike_price"]
    combined = combined.astype({"moneyness": "Float32"})

    # 3. Compute log moneyness
    combined["log_moneyness"] = np.log(
        combined["ForwardPrice"] / combined["strike_price"]
    )
    combined = combined.astype({"log_moneyness": "Float32"})

    # 4. (paper) define ATM, OTM, ITM options
    #     - OTM: $\frac{F_t}{K} < 0.97$
    #     - ATM: $0.97 \leq \frac{F_t}{K} < 1.03$
    #     - ITM: $\frac{F_t}{K} \geq 1.03$
    is_otm_mask = combined["moneyness"] < 0.97
    combined.loc[is_otm_mask, "op_level"] = "otm"

    is_atm_mask = (combined["moneyness"] >= 0.97) & (combined["moneyness"] < 1.03)
    combined.loc[is_atm_mask, "op_level"] = "atm"

    is_itm_mask = combined["moneyness"] >= 1.03
    combined.loc[is_itm_mask, "op_level"] = "itm"

    combined = combined.astype({"op_level": "category"})

    # 5. Compute black's option price
    combined["black_price"] = combined.apply(
        lambda row: black(
            row["cp_flag"].lower(),
            row["ForwardPrice"],
            row["strike_price"],
            row["years_to_expiry"],
            row["rate"],
            row["prev_vix"],
        ),
        axis=1,
    )
    combined = combined.astype({"black_price": "Float32"})

    # Check for rows with nulls
    print(
        "Check for rows with null values:", "\n", combined[combined.isna().any(axis=1)]
    )

    # 6. (paper) Scale the Black's option price
    combined["scaled_black_price"] = combined["black_price"] / combined["strike_price"]
    combined = combined.astype({"scaled_black_price": "Float32"})

    # 7. (paper) Scale the Market option price
    combined["scaled_market_price"] = combined["mid_price"] / combined["strike_price"]
    combined = combined.astype({"scaled_market_price": "Float32"})

    combined
    return (combined,)


@app.cell(hide_code=True)
def plot_volatility_smile_moneyness(combined):
    sns.set_theme(style="whitegrid", context="talk")

    plt.figure(figsize=(14, 8))  # Slightly larger for detail

    sns.scatterplot(
        data=combined,
        x="moneyness",
        y="impl_volatility",
        hue="cp_flag",
        alpha=0.4,
        palette={
            "C": "#2ecc71",
            "P": "#e74c3c",
        },
        edgecolor="w",
        linewidth=0.3,
        s=40,
    )

    plt.axvline(
        x=1, color="#34495e", linestyle="--", linewidth=1.5, label="At-The-Money"
    )

    sns.despine(left=True, bottom=True)

    plt.title("Implied Volatility Smile", fontsize=20, weight="bold", pad=20)
    plt.xlabel("Moneyness ($F/K$)", fontsize=14, labelpad=10)
    plt.ylabel("Implied Volatility", fontsize=14, labelpad=10)

    plt.legend(
        title="Option Type",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=2,
        frameon=False,
    )

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def plot_volatility_smile_log_moneyness(combined):
    sns.set_theme(style="whitegrid", context="talk")

    plt.figure(figsize=(14, 8))  # Slightly larger for detail

    sns.scatterplot(
        data=combined,
        x="log_moneyness",
        y="impl_volatility",
        hue="cp_flag",
        alpha=0.4,
        palette={
            "C": "#2ecc71",
            "P": "#e74c3c",
        },
        edgecolor="w",
        linewidth=0.3,
        s=40,
    )

    plt.axvline(
        x=0, color="#34495e", linestyle="--", linewidth=1.5, label="At-The-Money"
    )

    sns.despine(left=True, bottom=True)

    plt.title("Implied Volatility Smile", fontsize=20, weight="bold", pad=20)
    plt.xlabel("Log-Moneyness ($Ln(F/K)$)", fontsize=14, labelpad=10)
    plt.ylabel("Implied Volatility", fontsize=14, labelpad=10)

    plt.legend(
        title="Option Type",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=2,
        frameon=False,
    )

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def keep_atm_md():
    mo.md(r"""
    Filter steps:
    1. (paper) filter out options with moneyness > 1.1 or < 0.9
    """)
    return


@app.cell
def keep_atm(combined):
    # 1. (paper) filter out options with moneyness > 1.1 or < 0.9
    print(
        "DF size before filtering moneyness > 1.1 or < 0.9: ",
        len(combined),
        "\n",
    )

    moneyness_mask = (combined["moneyness"] > 1.1) | (combined["moneyness"] < 0.9)

    print(
        "DF size of options with moneyness > 1.1 or < 0.9: ",
        len(combined[moneyness_mask]),
        "\n",
    )

    combined1 = combined.loc[~moneyness_mask].copy()

    print(
        "DF size after filtering moneyness > 1.1 or < 0.9: ",
        len(combined1),
        "\n",
    )

    combined1
    return (combined1,)


@app.cell(hide_code=True)
def plot_volatility_smile_moneyness_filter(combined1):
    sns.set_theme(style="whitegrid", context="talk")

    plt.figure(figsize=(14, 8))  # Slightly larger for detail

    sns.scatterplot(
        data=combined1,
        x="moneyness",
        y="impl_volatility",
        hue="cp_flag",
        alpha=0.4,
        palette={
            "C": "#2ecc71",
            "P": "#e74c3c",
        },
        edgecolor="w",
        linewidth=0.3,
        s=40,
    )

    plt.axvline(
        x=1, color="#34495e", linestyle="--", linewidth=1.5, label="At-The-Money"
    )

    sns.despine(left=True, bottom=True)

    plt.title("Implied Volatility Smile", fontsize=20, weight="bold", pad=20)
    plt.xlabel("Moneyness ($F/K$)", fontsize=14, labelpad=10)
    plt.ylabel("Implied Volatility", fontsize=14, labelpad=10)

    plt.legend(
        title="Option Type",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=2,
        frameon=False,
    )

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def plot_volatility_smile_log_moneyness_filter(combined1):
    sns.set_theme(style="whitegrid", context="talk")

    plt.figure(figsize=(14, 8))  # Slightly larger for detail

    sns.scatterplot(
        data=combined1,
        x="log_moneyness",
        y="impl_volatility",
        hue="cp_flag",
        alpha=0.4,
        palette={
            "C": "#2ecc71",
            "P": "#e74c3c",
        },
        edgecolor="w",
        linewidth=0.3,
        s=40,
    )

    plt.axvline(
        x=0, color="#34495e", linestyle="--", linewidth=1.5, label="At-The-Money"
    )

    sns.despine(left=True, bottom=True)

    plt.title("Implied Volatility Smile", fontsize=20, weight="bold", pad=20)
    plt.xlabel("Log-Moneyness ($Ln(F/K)$)", fontsize=14, labelpad=10)
    plt.ylabel("Implied Volatility", fontsize=14, labelpad=10)

    plt.legend(
        title="Option Type",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=2,
        frameon=False,
    )

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def export_md():
    mo.md(r"""
    # Export
    """)
    return


@app.cell
def export(combined1):
    df_export = pl.from_pandas(
        (
            combined1.sort_values(
                by=["date", "cp_flag", "exdate", "strike_price"]
            ).reset_index(drop=True)
        ),
    )

    df_export.write_parquet(DATA_DIR / f"{START_DATE}_{END_DATE}_cleaned_data.parquet")

    df_export

    return


if __name__ == "__main__":
    app.run()
