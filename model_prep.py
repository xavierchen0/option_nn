import marimo

app = marimo.App()


with app.setup():
    import copy
    from pathlib import Path

    import joblib
    import marimo as mo
    import optuna
    import polars as pl
    import torch
    from sklearn.model_selection import train_test_split
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from tqdm.auto import tqdm

    DATA_DIR = Path("data")
    DATA_DIR.mkdir(exist_ok=True)

    DEVICE = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"Using {DEVICE} device")

    BATCH_SIZE = 512
    EPOCH_NUM = 1000

    START_DATE = "2025-07-01"
    END_DATE = "2025-08-31"


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
    df = pl.read_parquet(f"data/{START_DATE}_{END_DATE}_cleaned_data.parquet")

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

    X_train_val, y_train_val, K_train_val = (
        X_tmp.to_torch(return_type="tensor", dtype=pl.Float32),
        y_tmp.to_torch(return_type="tensor", dtype=pl.Float32),
        K_tmp.to_torch(return_type="tensor", dtype=pl.Float32),
    )

    train_dataset = TensorDataset(X_train, y_train, K_train)
    val_dataset = TensorDataset(X_val, y_val, K_val)
    test_dataset = TensorDataset(X_test, y_test, K_test)
    train_val_dataset = TensorDataset(X_train_val, y_train_val, K_train_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    mo.vstack(ui_elems1, align="stretch")

    return (
        train_loader,
        val_loader,
        test_loader,
        train_dataset,
        val_dataset,
        test_dataset,
        train_val_dataset,
    )


@app.cell(hide_code=True)
def define_model_md():
    mo.md(r"""
    # Define Hybird Model Version 1
    """)


@app.class_definition
class HybridModelV1(nn.Module):
    def __init__(self, n_layers, n_units, dropout_rate):
        super().__init__()

        layers = []
        input_dim = 5

        for _ in range(n_layers):
            layers.append(nn.Linear(input_dim, n_units))
            layers.append(nn.Sigmoid())
            layers.append(nn.Dropout(dropout_rate))
            input_dim = n_units

        self.layer_stack = nn.Sequential(*layers)

        self.output_layer = nn.Linear(input_dim, 1)

    def forward(self, x):
        features = self.layer_stack(x)

        out = self.output_layer(features)

        return out


@app.function
def train_model(training_specs, model, train_loader, val_loader):
    model = model(
        training_specs["n_layers"],
        training_specs["n_units"],
        training_specs["dropout_rate"],
    ).to(training_specs["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=training_specs["lr"])
    criterion = nn.L1Loss()

    train_loss_history = []
    val_loss_history = []

    best_val_loss = float("inf")
    best_model_state = copy.deepcopy(model.state_dict())

    print("=================================================")
    print(f"Starting training on {training_specs['device']}")

    epoch_pbar = tqdm(range(training_specs["epoch_num"]), desc="Training Progress")

    for epoch in epoch_pbar:
        # Training
        model.train()
        train_loss = 0.0

        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False)

        for batch_X, batch_y, _ in batch_pbar:
            batch_X, batch_y = (
                batch_X.to(training_specs["device"]),
                batch_y.to(training_specs["device"]),
            )

            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_X.size(0)

        epoch_train_loss = train_loss / len(train_loader.dataset)
        train_loss_history.append(epoch_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_X, batch_y, _ in val_loader:
                batch_X, batch_y = (
                    batch_X.to(training_specs["device"]),
                    batch_y.to(training_specs["device"]),
                )

                preds = model(batch_X)
                loss = criterion(preds, batch_y)
                val_loss += loss.item() * batch_X.size(0)

        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_loss_history.append(epoch_val_loss)

        # Checkpointing
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = copy.deepcopy(model.state_dict())

        epoch_pbar.set_postfix(
            {
                "Train MAE": f"{epoch_train_loss:.5f}",
                "Val MAE": f"{epoch_val_loss:.5f}",
                "Best Val": f"{best_val_loss:.5f}",
            }
        )

    print("=================================================")
    print("Training Complete")
    torch.save(
        best_model_state,
        training_specs["data_dir"]
        / f"{training_specs['start_date']}_{training_specs['end_date']}_{training_specs['output_filename']}_weights.pt",
    )

    loss_df = pl.DataFrame({"train": train_loss_history, "val": val_loss_history})

    loss_df.write_parquet(
        training_specs["data_dir"]
        / f"{training_specs['start_date']}_{training_specs['end_date']}_{training_specs['output_filename']}_loss_history.parquet"
    )


@app.cell(hide_code=True)
def optuna_objective_md():
    mo.md(r"""
    # Define Optuna Objective
    """)


@app.cell
def optuna_objective(train_loader, val_loader):
    def objective(trial):
        print("========================================================")
        print(f"Executing Trial {trial.number}")

        # Optuna suggest values
        n_layers = trial.suggest_int("n_layers", 1, 6)
        n_units = trial.suggest_int("n_units", 20, 100)
        dropout_rate = trial.suggest_float("dropout_rate", 0.01, 0.2)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

        # Model specification
        model = HybridModelV1(n_layers, n_units, dropout_rate).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.L1Loss()

        # Model optimisation
        epoch_tune = 20
        for epoch in range(epoch_tune):
            model.train()

            for batch_X, batch_y, _ in train_loader:
                batch_X = batch_X.to(DEVICE)
                batch_y = batch_y.to(DEVICE)

                optimizer.zero_grad()
                prediction = model(batch_X)
                loss = criterion(prediction, batch_y)
                loss.backward()
                optimizer.step()

            # Optuna Validation and pruning
            model.eval()
            val_loss = 0.0
            steps = 0

            with torch.no_grad():
                for batch_X, batch_y, _ in val_loader:
                    batch_X = batch_X.to(DEVICE)
                    batch_y = batch_y.to(DEVICE)

                    prediction = model(batch_X)
                    val_loss += criterion(prediction, batch_y).item()
                    steps += 1

            avg_val_loss = val_loss / steps

            trial.report(avg_val_loss, epoch)

            if trial.should_prune():
                raise optuna.TrialPruned()

        return avg_val_loss

    return objective


@app.cell(hide_code=True)
def tune_hyper_md():
    mo.md(r"""
    # Hyperparameter Optimisation
    1. `n_layers`: number of layers
    2. `n_units`: number of neurons per layer
    3. `dropout_rate`: during training, randomly zeroes some of the elements of the input tensor with probability of `dropout_rate`
    4. `lr`: learning rate
    """)


@app.cell
def tune_hyper(objective):
    study = optuna.create_study(direction="minimize", study_name="option_nn")

    study.optimize(objective, n_trials=50, n_jobs=-1)

    return study


@app.cell(hide_code=True)
def tune_hyper_res_md():
    mo.md(r"""
    # Hyperparameter Optimisation Results
    """)


@app.cell
def tune_hyper_res(study):
    mo.vstack(
        [
            mo.md(f"Best params: {study.best_params}"),
            mo.md(f"Best avg val loss: {study.best_value}"),
            optuna.visualization.plot_optimization_history(study),
            optuna.visualization.plot_timeline(study),
            optuna.visualization.plot_param_importances(study),
        ],
        align="stretch",
    )


@app.cell(hide_code=True)
def export_md():
    mo.md(r"""
    # Export study
    """)


@app.cell
def export(study, train_dataset, val_dataset, test_dataset, train_val_dataset):
    joblib.dump(study, DATA_DIR / f"{START_DATE}_{END_DATE}_study.pkl")

    torch.save(train_dataset, DATA_DIR / f"{START_DATE}_{END_DATE}_train.pt")
    torch.save(val_dataset, DATA_DIR / f"{START_DATE}_{END_DATE}_val.pt")
    torch.save(test_dataset, DATA_DIR / f"{START_DATE}_{END_DATE}_test.pt")
    torch.save(train_val_dataset, DATA_DIR / f"{START_DATE}_{END_DATE}_train_val.pt")


if __name__ == "__main__":
    app.run()
