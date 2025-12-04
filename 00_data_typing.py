import marimo

__generated_with = "0.18.1"
app = marimo.App()

with app.setup:
    from params import PANDAS_SCHEMA, read_csv_with_schema
    from pathlib import Path

    import polars as pl
    import pandas as pd

    DATA_DIR = Path("data")
    DATA_DIR.mkdir(exist_ok=True)


@app.cell
def inspect_data():
    # q_test = pl.scan_csv(DATA_DIR / "01_raw" / "options.csv")

    # Explore values in columns to determine appropriate typing
    # for c in [...]:
    #     print(
    #         q_test.select([c])
    #         .unique()
    #         .collect()
    #     )
    return


@app.cell
def convert_csv_to_parquet():
    if True:
        FILES = ["options", "interest", "forwards", "vix"]
        csv_dir = DATA_DIR / "01_raw"
        output_dir = DATA_DIR / "02_intermediate"
        output_dir.mkdir(exist_ok=True)

        for f in FILES:
            csv_path = csv_dir / f"{f}.csv"
            output_path = output_dir / f"{f}.parquet"

            df = read_csv_with_schema(csv_path, PANDAS_SCHEMA)
            df.to_parquet(output_path)
    return


@app.cell
def get_unique_option_combinations():
    if True:
        (
            pl.scan_parquet(DATA_DIR / "02_intermediate" / "options.parquet")
            .select(["date", "exdate", "am_settlement"])
            .unique()
            .sort("date", "exdate", "am_settlement")
            .collect()
            .write_parquet(DATA_DIR / "03_processed" / "opt_combs.parquet")
        )
    return


if __name__ == "__main__":
    app.run()
