from lets_plot import gggrid
import marimo
from sqlalchemy import ResultProxy
from sqlalchemy.engine import result

app = marimo.App()

with app.setup():
    from pathlib import Path

    import lets_plot as lp
    import marimo as mo
    import polars as pl
    from lets_plot import ggplot
    from py_vollib.black import black
    from pysabr import Hagan2002LognormalSABR, hagan_2002_lognormal_sabr
    from tqdm.auto import tqdm

    DATA_DIR = Path("data")
    DATA_DIR.mkdir(exist_ok=True)

    # START_DATE = "2025-07-01"
    # END_DATE = "2025-08-31"
    START_DATE = "2020-09-01"
    END_DATE = "2025-08-29"

    df = pl.read_parquet(DATA_DIR / f"{START_DATE}_{END_DATE}_cleaned_data.parquet")


@app.cell(hide_code=True)
def preview_date_md():
    mo.md(r"""
    # Preview data
    """)


@app.cell(hide_code=True)
def preview_data():
    df


@app.cell(hide_code=True)
def calibrate_sabr_md():
    mo.md(r"""
    # Calibrate SABR

    Steps:
    1. Prepare inputs for SABR model
        - Forward Price
        - Time to expiry in years
        - Implied volatility (IvyDB)
        - Option strikes
        - BETA in SABR
    2. Calibrate SABR
    3. Calculate SABR volatility for all strikes in each group
    4. Calculate SABR option price
    """)


@app.function()
def calibrate_sabr(group):
    # 1. Prepare inputs for SABR model
    f = group.item(0, "ForwardPrice")
    t = group.item(0, "years_to_expiry")
    iv = group["impl_volatility"].to_numpy() * 100
    strikes = group["strike_price"].to_numpy()
    beta = 0.8

    # 2. Calibrate SABR
    sabr = Hagan2002LognormalSABR(f=f, t=t, shift=0, beta=beta)

    alpha, rho, volvol = sabr.fit(strikes, iv)

    sabr_vols = [
        hagan_2002_lognormal_sabr.lognormal_vol(k, f, t, alpha, beta, rho, volvol)
        for k in strikes
    ]

    return group.with_columns(
        [
            pl.lit(alpha, dtype=pl.Float32).alias("sabr_alpha"),
            pl.lit(rho, dtype=pl.Float32).alias("sabr_rho"),
            pl.lit(volvol, dtype=pl.Float32).alias("sabr_volvol"),
            pl.Series(sabr_vols, dtype=pl.Float32).alias("sabr_vol"),
        ]
    )


@app.cell
def compute_sabr():
    groups = df.partition_by(["date", "exdate"])

    # 3. Calculate SABR volatility for all strikes in each group
    results = [
        calibrate_sabr(group)
        for group in tqdm(
            groups, desc="Finding SABR params for each group", total=len(groups)
        )
    ]

    combined = pl.concat(results)

    # 4. Calculate SABR option price
    combined = combined.with_columns(
        pl.struct(
            [
                "cp_flag",
                "ForwardPrice",
                "strike_price",
                "years_to_expiry",
                "rate",
                "sabr_vol",
            ]
        )
        .map_elements(
            lambda row: black(
                row["cp_flag"].lower(),
                row["ForwardPrice"],
                row["strike_price"],
                row["years_to_expiry"],
                row["rate"],
                row["sabr_vol"],
            ),
            return_dtype=pl.Float32,
        )
        .alias("sabr_price")
    )

    check_null_df = combined.filter(pl.any_horizontal(pl.all().is_null()))

    combined.sort(["date", "cp_flag", "exdate", "strike_price"]).write_parquet(
        DATA_DIR / f"{START_DATE}_{END_DATE}_cleaned_data.parquet"
    )

    mo.vstack(
        [mo.md(r"Check null:"), check_null_df, mo.md("Combined:"), combined],
        align="stretch",
    )

    return combined


@app.cell(hide_code=True)
def plot_chart_md():
    mo.md(r"""
    # Plot chart

    Use the date and exdate with the most number of options.
    """)


@app.cell(hide_code=True)
def plot_chart_iv(combined):
    largest_group = (
        combined.group_by(["date", "exdate"]).len().sort("len", descending=True)
    ).head(1)

    largest_date, largest_exdate = (
        largest_group.item(0, "date"),
        largest_group.item(0, "exdate"),
    )

    largest_options = combined.filter(
        (pl.col("date") == largest_date) & (pl.col("exdate") == largest_exdate)
    )

    largest_options_calls = largest_options.filter(pl.col("cp_flag") == "C")
    largest_options_puts = largest_options.filter(pl.col("cp_flag") == "P")

    largest_options_calls_iv = largest_options_calls.unpivot(
        on=["sabr_vol", "impl_volatility"],
        index="strike_price",
        variable_name="source_of_vol",
        value_name="impl_volatility",
    )
    largest_options_puts_iv = largest_options_puts.unpivot(
        on=["sabr_vol", "impl_volatility"],
        index="strike_price",
        variable_name="source_of_vol",
        value_name="impl_volatility",
    )

    sabr_iv_chart = lp.gggrid(
        [
            ggplot(
                chart_df,
                lp.aes(x="strike_price", y="impl_volatility", color="source_of_vol"),
            )
            + lp.geom_point()
            + lp.geom_line()
            + lp.labs(x="Strike Price", y="Implied Volatility", color="Source")
            + lp.ggtitle(
                f"{op_type} on {largest_date.strftime('%Y-%m-%d')} and expiring on {largest_exdate.strftime('%Y-%m-%d')}"
            )
            for chart_df, op_type in [
                (largest_options_calls_iv, "Calls"),
                (largest_options_puts_iv, "Puts"),
            ]
        ],
        ncol=1,
    ) + lp.ggtitle("SABR IV")

    largest_options_calls_price = largest_options_calls.unpivot(
        on=["sabr_price", "mid_price"],
        index="strike_price",
        variable_name="source_of_price",
        value_name="price",
    )
    largest_options_puts_price = largest_options_puts.unpivot(
        on=["sabr_price", "mid_price"],
        index="strike_price",
        variable_name="source_of_price",
        value_name="price",
    )

    sabr_price_chart = lp.gggrid(
        [
            ggplot(
                chart_df,
                lp.aes(x="strike_price", y="price", color="source_of_price"),
            )
            + lp.geom_point()
            + lp.geom_line()
            + lp.labs(x="Strike Price", y="Price", color="Source")
            + lp.ggtitle(
                f"{op_type} on {largest_date.strftime('%Y-%m-%d')} and expiring on {largest_exdate.strftime('%Y-%m-%d')}"
            )
            for chart_df, op_type in [
                (largest_options_calls_price, "Calls"),
                (largest_options_puts_price, "Puts"),
            ]
        ],
        ncol=1,
    ) + lp.ggtitle("SABR Price")

    mo.vstack(
        [
            mo.md(f"Largest date: {largest_date}"),
            mo.md(f"Largest exdate: {largest_exdate}"),
            mo.md(r"DF of largest date and largest exdate:"),
            largest_options.select(
                "impl_volatility", "sabr_vol", "ForwardPrice", "strike_price"
            ),
            mo.md(r"IV Chart:"),
            sabr_iv_chart,
            mo.md(r"Price Chart:"),
            sabr_price_chart,
        ],
        align="stretch",
    )


if __name__ == "__main__":
    app.run()
