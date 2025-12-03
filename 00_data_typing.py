import marimo

__generated_with = "0.18.1"
app = marimo.App()

with app.setup:
    from params import POLARS_SCHEMA
    from pathlib import Path

    import polars as pl

    DATA_DIR = Path("data")
    DATA_DIR.mkdir(exist_ok=True)


@app.cell
def scan_data():
    # q_test = pl.scan_csv(DATA_DIR / "01_raw" / "options.csv")
    return


@app.cell
def inspect_data():
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
    def csv_to_parquet():
        FILES = ["options", "interest", "forwards", "vix"]
        csv_dir = DATA_DIR / "01_raw"
        output_dir = DATA_DIR / "02_intermediate"
        output_dir.mkdir(exist_ok=True)

        for f in FILES:
            csv_path = csv_dir / f"{f}.csv"
            output_path = output_dir / f"{f}.parquet"

            q_csv = pl.scan_csv(csv_path, schema_overrides=POLARS_SCHEMA)
            q_csv.sink_parquet(output_path)

    csv_to_parquet()
    return


@app.cell
def base_query():
    q_base = pl.scan_parquet(DATA_DIR / "02_intermediate" / "options.parquet")
    return (q_base,)


@app.cell
def get_unique_option_combinations(q_base):
    def get_option_combinations():
        q_combs = q_base.select(["date", "exdate", "am_settlement"]).unique().sort("date", "exdate", "am_settlement")

        q_combs.collect().write_parquet(DATA_DIR / "03_processed" / "opt_combs.parquet")

    return


if __name__ == "__main__":
    app.run()
