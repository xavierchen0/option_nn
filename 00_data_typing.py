import marimo

__generated_with = "0.18.1"
app = marimo.App()


@app.cell
def _():
    import polars as pl
    import os
    from pathlib import Path
    from params import OPTION_SCHEMA
    return OPTION_SCHEMA, Path, os, pl


@app.cell
def _():
    # q_test = pl.scan_csv("data/01_raw/options.csv")
    return


@app.cell
def _():
    # Explore values in columns to determine appropriate typing
    # for c in [...]:
    #     print(
    #         q_test.select([c])
    #         .unique()
    #         .collect()
    #     )
    return


@app.cell
def _(OPTION_SCHEMA, Path, os, pl):
    def csv_to_parquet():

        csv_path = "data/01_raw/options.csv"
        output_path = Path("data/02_intermediate/options_all.parquet")
        os.makedirs(output_path.parent, exist_ok=True)

        q_csv = pl.scan_csv(csv_path, schema_overrides=OPTION_SCHEMA)
        q_csv.sink_parquet(output_path)

    csv_to_parquet()
    return


@app.cell
def _(pl):
    q_base = pl.scan_parquet("data/02_intermediate/options_all.parquet")
    return (q_base,)


@app.cell
def _(q_base):
    def get_option_combinations():

        q_combs = \
            (
            q_base
            .select(["date", "exdate", "am_settlement"]).unique()
            .sort("date", "exdate", "am_settlement")
            )

        q_combs.collect().write_parquet("data/03_processed/opt_combs.parquet")

    get_option_combinations()
    return


if __name__ == "__main__":
    app.run()
