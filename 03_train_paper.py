import marimo

app = marimo.App()

with app.setup:
    from pathlib import Path

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

    # Training specifications
    BATCH_SIZE = 512
    EPOCH_NUM = 1000
    LR = 0.001

    # Model Hyperparameters
    N_LAYERS = 2
    N_UNITS = 40
    DROPOUT_RATE = 0.1

    # Output filename for the weights
    OUTPUT_FILENAME = "paper"

    # Data date range
    START_DATE = "2025-07-01"
    END_DATE = "2025-08-31"

    training_specs = {}
    training_specs["n_layers"] = N_LAYERS
    training_specs["n_units"] = N_UNITS
    training_specs["dropout_rate"] = DROPOUT_RATE
    training_specs["device"] = DEVICE
    training_specs["lr"] = LR
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
def train():
    train_dataset = torch.load(
        DATA_DIR / f"{START_DATE}_{END_DATE}_train.pt", weights_only=False
    )
    val_dataset = torch.load(
        DATA_DIR / f"{START_DATE}_{END_DATE}_val.pt", weights_only=False
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    train_model(training_specs, HybridModelV1, train_loader, val_loader)


if __name__ == "__main__":
    app.run()
