import marimo

__generated_with = "0.18.1"
app = marimo.App()

with app.setup:
    from params import POLARS_SCHEMA
    from pathlib import Path

    import lets_plot as lp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import polars as pl
    import QuantLib as ql
    import seaborn as sns

    lp.LetsPlot.setup_html()

    START_DATE = pd.Timestamp("2025-07-01")
    END_DATE = pd.Timestamp("2025-08-31")

    DATA_DIR = Path("data")
    DATA_DIR.mkdir(exist_ok=True)


@app.cell(hide_code=True)
def data_sources():
    mo.md(r"""
    # Data

    #### Query Data Range
    - 29 Aug 2020 - 29 Aug 2025
    - All European options

    #### Data Sources
    - Options data is queried from OptionMetrics through WRDS [link](https://wrds-www.wharton.upenn.edu/pages/get-data/optionmetrics/ivy-db-us/options/option-prices/)
    - Forward Prices data is queried from OptionMetrics through WRDS [link](https://wrds-www.wharton.upenn.edu/pages/get-data/optionmetrics/ivy-db-us/options/forward-price/)
    - Interest Rates data is queried from OptionMetrics through WRDS [link](https://wrds-www.wharton.upenn.edu/pages/get-data/optionmetrics/ivy-db-us/market/zero-coupon-yield-curve/)
    - VIX data is queried from CBOE Options through WRDS [link](https://wrds-www.wharton.upenn.edu/pages/get-data/cboe-indexes/cboe-indexes-1/cboe-indexes/)
    """)
    return


@app.cell(hide_code=True)
def read_md():
    mo.md(r"""
    # Read datasets
    """)
    return


@app.cell
def read_data():
    df_options = pl.read_parquet("data/02_intermediate/options.parquet").to_pandas(use_pyarrow_extension_array=True)
    df_forwards = pl.read_parquet("data/02_intermediate/forwards.parquet").to_pandas(use_pyarrow_extension_array=True)
    df_interest = pl.read_parquet("data/02_intermediate/interest.parquet").to_pandas(use_pyarrow_extension_array=True)
    df_vix = pl.read_parquet("data/02_intermediate/vix.parquet").to_pandas(use_pyarrow_extension_array=True)
    return df_forwards, df_interest, df_options, df_vix


@app.cell(hide_code=True)
def clean_data_md():
    mo.md(r"""
    ## Data Cleaning and Preprocessing

    ##### I. Options Data
    1. Keep necessary columns
    2. Filter for specific dates
    3. Remove rows with non-standard expirations
    4. Remove rows with null IVs due to special settlement type
    5. Divide strike by 1000; In OptionMetric, strike is x 1000
    6. (paper) Filter out no open interest and no volume
    7. Filter out best_bid == 0 as they are not representative
    8. Calculate mid option price
    9. (paper) Filter mid_price < 1/8
    10. Calculate days to expiry, adjusted for am_settlement
    11. (paper) Filter days_to_expiry > 120

    ##### II. Forwards Data
    12. Merge necessary columns from forwards
    13. Compute moneyness $\frac{F_t}{K}$ and log moneyness $\log \frac{F_t}{K}$
    14. (paper) filter out options with moneyness > 1.1 or < 0.9

    ##### III. Interest Rates Data
    15. Using QuantLib's MonotonicCubicZeroCurve, fit a Monotonic Cubic Spline across zero rates at each date
    16. With reference to option date and days to expiry, compute the respective risk-free rate $r$.

    ##### IV. VIX Data
    17. Merge necessary columns from VIX
    18. Convert VIX index from percentage to decimal
    """)
    return


@app.cell
def clean_data_i(df_options):
    # 1. Keep necessary columns
    print("1. Keep necessary columns")
    df_processed_i = df_options[
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
            "ss_flag",
        ]
    ].copy()

    print("Option's columns and types:")
    print(df_processed_i.dtypes)
    print("DF shape initially: ", df_processed_i.shape)
    print("")

    # 2. Filter for specific dates
    print("2. Filter for specific dates")

    options_date_mask = (df_processed_i["date"] >= START_DATE) & (df_processed_i["date"] <= END_DATE)
    df_processed_i = df_processed_i[options_date_mask].copy()

    print("Maximum date: ", df_processed_i["date"].max())
    print("Minimum date: ", df_processed_i["date"].min())
    print("DF shape after filtering for dates: ", df_processed_i.shape)
    print("")

    # 3. Remove rows with non-standard expirations
    print("3. Remove rows with non-standard expirations")

    ss_mask = df_processed_i["ss_flag"] != "E"
    df_processed_i = df_processed_i[ss_mask].drop(columns=["ss_flag"]).copy()

    print("DF shape after dropping non-standard expirations: ", df_processed_i.shape)
    print("")

    # 4. Remove rows with null IVs
    print("4. Remove rows with null IVs")

    df_processed_i = df_processed_i.dropna(axis=0, subset=["impl_volatility"])

    print("DF shape after dropping null IV: ", df_processed_i.shape)
    print("")

    # 5. Divide strike by 1000; In OptionMetric, strike is x 1000
    print("5. Divide strike by 1000; In OptionMetric, strike is x 1000")

    df_processed_i["strike_price"] = df_processed_i["strike_price"] / 1000

    print("DF shape after changing strike: ", df_processed_i.shape)
    print("")

    # 6. (paper) filter out no open interest and no volume
    print("6. (paper) filter out no open interest and no volume")

    no_open_interest_volume_mask = (df_processed_i["open_interest"] == 0) | (df_processed_i["volume"] == 0)
    df_processed_i = df_processed_i.loc[~no_open_interest_volume_mask].drop(columns=["open_interest", "volume"]).copy()

    print("DF shape after filtering out no open interest and no volume: ", df_processed_i.shape)
    print("")

    # 7. filter out best_bid == 0 as they are not representative
    print("7. filter out best_bid == 0 as they are not representative")

    best_bid_0_mask = df_processed_i["best_bid"] == 0
    df_processed_i = df_processed_i.loc[~best_bid_0_mask].copy()

    print("DF shape after filtering out best_bid == 0: ", df_processed_i.shape)
    print("")

    # 8. calculate mid option price
    print("8. calculate mid option price")

    df_processed_i["mid_price"] = (df_processed_i["best_bid"] + df_processed_i["best_offer"]) / 2
    df_processed_i = df_processed_i.drop(columns=["best_bid", "best_offer"])

    print("DF shape after adding mid_price: ", df_processed_i.shape)
    print("")

    # 9. (paper) filter mid_price < 1/8
    print("9. (paper) filter mid_price < 1/8")

    mid_price_less_one_eighth_mask = df_processed_i["mid_price"] < 0.125
    df_processed_i = df_processed_i[~mid_price_less_one_eighth_mask].copy()

    print("DF shape after filtering out mid_price < 1/8: ", df_processed_i.shape)
    print("")

    # 10. Calculate days to expiry, adjusted for am_settlement
    print("10. Calculate days to expiry, adjusted for am_settlement")

    df_processed_i["days_to_expiry"] = (df_processed_i["exdate"] - df_processed_i["date"]).dt.days + np.where(
        df_processed_i["am_settlement"], -1, 0
    )
    df_processed_i = df_processed_i.astype({"days_to_expiry": "Int64"})

    print("DF shape after adding days_to_expiry: ", df_processed_i.shape)
    print("")

    # 11. (paper) Filter days_to_expiry > 120
    print("11. (paper) Filter days_to_expiry > 120")

    days_to_expiry_more_120_mask = df_processed_i["days_to_expiry"] > 120
    df_processed_i = df_processed_i.loc[~days_to_expiry_more_120_mask].copy()

    print("DF shape after filtering out days_to_expiry > 120: ", df_processed_i.shape)
    print("")
    return (df_processed_i,)


@app.cell
def clean_data_ii(df_forwards, df_processed_i):
    # 12. Merge necessary columns from forwards
    print("12. Merge necessary columns from forwards")

    forwards_tmp = df_forwards[["date", "ForwardPrice", "expiration", "AMSettlement"]].copy()
    df_processed_ii = (
        df_processed_i.merge(
            forwards_tmp,
            left_on=["date", "exdate", "am_settlement"],
            right_on=["date", "expiration", "AMSettlement"],
            how="inner",
        )
        .reset_index(drop=True)
        .drop(columns=["expiration", "am_settlement", "AMSettlement"])
        .sort_values(by="date")
    )

    print("DF shape after merging forwards: ", df_processed_ii.shape)
    print("")

    # 13. Compute moneyness Ft/K and log moneyness log Ft/K
    print("13. Compute moneyness Ft/K and log moneyness log Ft/K")

    df_processed_ii["moneyness"] = df_processed_ii["ForwardPrice"] / df_processed_ii["strike_price"]
    df_processed_ii["log_moneyness"] = np.log(df_processed_ii["ForwardPrice"] / df_processed_ii["strike_price"])

    print("DF shape after computing moneyness and log moneyness: ", df_processed_ii.shape)
    print("")

    # 14. (paper) filter out options with moneyness > 1.1 or < 0.9
    print("14. (paper) filter out options with moneyness > 1.1 or < 0.9")

    moneyness_mask = (df_processed_ii["moneyness"] > 1.1) | (df_processed_ii["moneyness"] < 0.9)
    df_processed_ii = df_processed_ii[~moneyness_mask].copy()

    print("DF shape after computing moneyness and log moneyness: ", df_processed_ii.shape)
    return (df_processed_ii,)


@app.function
def to_ql_date(d):
    return ql.Date(d.day, d.month, d.year)


@app.cell
def clean_data_iii(df_interest, df_processed_ii):
    # 15. Using QuantLib's MonotonicCubicZeroCurve, fit a Monotonic Cubic Spline across zero rates at each date
    print("15. Using QuantLib's MonotonicCubicZeroCurve, fit a Monotonic Cubic Spline across zero rates at each date")

    dates_idx = df_interest["date"].unique()
    zc_series = pd.Series(index=dates_idx, name="ZeroCurve", dtype="object")
    for curve_date in dates_idx:
        specific_date_mask = df_interest["date"] == curve_date
        df_target = df_interest[specific_date_mask].copy()
        base_rate = df_target.sort_values("date")["rate"].iloc[0]
        base_row = pd.DataFrame([{"date": curve_date, "days": 0, "rate": base_rate}])
        df_target = pd.concat([base_row, df_target], axis="rows").sort_values("days")
        df_target["maturity"] = (df_target["date"] + df_target["days"] * pd.Timedelta(days=1)).apply(to_ql_date)
        zc = ql.MonotonicCubicZeroCurve(
            df_target["maturity"].values,
            df_target["rate"],
            ql.Actual365Fixed(),
            ql.UnitedStates(ql.UnitedStates.GovernmentBond),
        )
        zc_series.loc[curve_date] = zc

    print(f"Unique dates in array of zero curves: {len(zc_series)}")
    print("")

    # 16. With reference to option date and days to expiry, compute the respective risk-free rate r.
    print("16. With reference to option date and days to expiry, compute the respective risk-free rate r.")

    def get_rate(current_date, days_to_expiry):
        return (
            zc_series.loc[current_date]
            .zeroRate(
                to_ql_date(current_date) + days_to_expiry,
                ql.Actual365Fixed(),
                ql.Continuous,
                ql.NoFrequency,
                True,
            )
            .rate()
        )

    df_processed_iii = df_processed_ii.copy()
    df_processed_iii["rate"] = df_processed_iii.apply(
        lambda x: get_rate(x["date"], x["days_to_expiry"]), axis="columns"
    )
    df_processed_iii = df_processed_iii.dropna(subset=["rate"])

    print("DF shape after adding rate: ", df_processed_iii.shape)
    return dates_idx, df_processed_iii, get_rate


@app.cell
def clean_data_iv(df_processed_iii, df_vix):
    # 17. Merge necessary columns from VIX
    print("17. Merge necessary columns from VIX")

    df_processed_iv = (
        df_processed_iii.merge(df_vix[["Date", "vix"]], left_on=["date"], right_on="Date", how="left")
        .reset_index(drop=True)
        .drop(columns=["Date"])
    )

    print("DF shape after adding VIX: ", df_processed_iv.shape)
    print("")

    # 18. Convert VIX index from percentage to decimal
    print("18. Convert VIX index from percentage to decimal")

    df_processed_iv["vix"] = df_processed_iv["vix"] / 100

    print("DF shape after transforming VIX: ", df_processed_iv.shape)
    return (df_processed_iv,)


@app.cell(hide_code=True)
def plot_rate_curve(dates_idx, get_rate):
    def plot_curve(date_str, period=pd.DateOffset(years=5)):
        matching_date = dates_idx == pd.Timestamp(date_str)
        if sum(matching_date) == 1:
            i = matching_date.argmax()
        else:
            raise ValueError("Invalid date argument")

        current_date = dates_idx[i]

        date_range = pd.date_range(current_date, freq="D", end=current_date + period)
        df_plot = pd.DataFrame(np.arange(len(date_range)), index=date_range, columns=["days_to_expiry"])
        df_plot["rate"] = df_plot.apply(lambda x: get_rate(current_date, x["days_to_expiry"]), axis="columns")

        # Plot
        p = (
            lp.ggplot(df_plot, lp.aes(x="days_to_expiry", y="rate"))
            + lp.geom_line(color="#3f51b5", size=1.2)
            + lp.ggtitle(f"Interpolated Zero Curve: {current_date}")
            + lp.labs(x="Days from Valuation", y="Zero Rate")
            + lp.theme_minimal()
        )

        return p

    # Render plot
    plot_curve("2020-08-31")
    return


@app.cell(hide_code=True)
def plot_volatility_smile_moneyness_filter(df_processed_iv):
    sns.set_theme(style="whitegrid", context="talk")

    plt.figure(figsize=(14, 8))  # Slightly larger for detail

    sns.scatterplot(
        data=df_processed_iv,
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

    plt.axvline(x=1, color="#34495e", linestyle="--", linewidth=1.5, label="At-The-Money")

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
def plot_volatility_smile_log_moneyness_filter(df_processed_iv):
    sns.set_theme(style="whitegrid", context="talk")

    plt.figure(figsize=(14, 8))  # Slightly larger for detail

    sns.scatterplot(
        data=df_processed_iv,
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

    plt.axvline(x=0, color="#34495e", linestyle="--", linewidth=1.5, label="At-The-Money")

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
def export(df_processed_iv):
    df_export = pl.from_pandas(
        (df_processed_iv.sort_values(by=["date", "cp_flag", "exdate", "strike_price"]).reset_index(drop=True)),
        schema_overrides=POLARS_SCHEMA,
    )

    df_export.write_parquet(DATA_DIR / "03_processed" / "cleaned_data.parquet")
    df_export
    return


if __name__ == "__main__":
    app.run()
