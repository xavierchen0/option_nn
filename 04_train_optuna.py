import marimo

app = marimo.App()

with app.setup:
    from pathlib import Path
    import pprint

    import joblib
    import marimo as mo
    import torch
    from torch.utils.data import DataLoader

    from model_prep import HybridModelV1, train_model

    DATA_DIR = Path("data")
    DATA_DIR.mkdir(exist_ok=True)

    DEVICE = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )

    # Data date range
    START_DATE = "2025-07-01"
    END_DATE = "2025-08-31"

    # Optuna
    study_calls = joblib.load(DATA_DIR / f"{START_DATE}_{END_DATE}_study_calls.pkl")
    study_puts = joblib.load(DATA_DIR / f"{START_DATE}_{END_DATE}_study_puts.pkl")

    print("Study (calls):", "\n", study_calls.best_params)
    print("Study (puts):", "\n", study_puts.best_params)

    # Training specifications
    BATCH_SIZE = 512
    EPOCH_NUM = 1000

    # Output filename for the weights
    OUTPUT_FILENAME = "optuna"

    training_specs = {}
    training_specs["device"] = DEVICE
    training_specs["epoch_num"] = EPOCH_NUM
    training_specs["data_dir"] = DATA_DIR
    training_specs["output_filename"] = OUTPUT_FILENAME
    training_specs["start_date"] = START_DATE
    training_specs["end_date"] = END_DATE


@app.cell(hide_code=True)
def train_md():
    mo.md(r"""
    # Train model
    """)


@app.cell
def train_calls():
    train_calls_dataset = torch.load(
        DATA_DIR / f"{START_DATE}_{END_DATE}_train_calls.pt", weights_only=False
    )
    val_calls_dataset = torch.load(
        DATA_DIR / f"{START_DATE}_{END_DATE}_val_calls.pt", weights_only=False
    )

    train_calls_loader = DataLoader(
        train_calls_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    val_calls_loader = DataLoader(val_calls_dataset, batch_size=BATCH_SIZE)

    training_specs["option_type"] = "calls"
    training_specs["n_layers"] = study_calls.best_params["n_layers"]
    training_specs["n_units"] = study_calls.best_params["n_units"]
    training_specs["dropout_rate"] = study_calls.best_params["dropout_rate"]
    training_specs["lr"] = study_calls.best_params["lr"]

    train_model(training_specs, HybridModelV1, train_calls_loader, val_calls_loader)


@app.cell
def train_puts():
    train_puts_dataset = torch.load(
        DATA_DIR / f"{START_DATE}_{END_DATE}_train_puts.pt", weights_only=False
    )
    val_puts_dataset = torch.load(
        DATA_DIR / f"{START_DATE}_{END_DATE}_val_puts.pt", weights_only=False
    )

    train_puts_loader = DataLoader(
        train_puts_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    val_puts_loader = DataLoader(val_puts_dataset, batch_size=BATCH_SIZE)

    training_specs["option_type"] = "puts"
    training_specs["n_layers"] = study_puts.best_params["n_layers"]
    training_specs["n_units"] = study_puts.best_params["n_units"]
    training_specs["dropout_rate"] = study_puts.best_params["dropout_rate"]
    training_specs["lr"] = study_puts.best_params["lr"]

    train_model(training_specs, HybridModelV1, train_puts_loader, val_puts_loader)


if __name__ == "__main__":
    app.run()
