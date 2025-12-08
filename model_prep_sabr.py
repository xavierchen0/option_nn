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

    # START_DATE = "2025-07-01"
    # END_DATE = "2025-08-31"
    START_DATE = "2020-09-01"
    END_DATE = "2025-08-29"


@app.cell(hide_code=True)
def prepare_data_md():
    mo.md(r"""
    # Prepare data

    Steps:
    1. Read input dataset via polars
    2. (paper) separate call and put options
    3. (paper) Train validation test split chronologically: 80% train, 20% test
    4. (paper) Train validation test split chronologically: 75% of train = train, 25% of train = validation
    5. Convert to torch
    """)


@app.cell
def prepare_data():
    # 1. Read input dataset via polars
    df = pl.read_parquet(f"data/{START_DATE}_{END_DATE}_cleaned_data.parquet")

    # 2. (paper) separate call and put options
    df_calls = df.filter(pl.col("cp_flag") == "C")
    df_puts = df.filter(pl.col("cp_flag") == "P")

    # 3. (paper) Train validation test split chronologically: 80% train, 20% test
    feature_cols = [
        "moneyness",
        "rate",
        "prev_vix",
        "years_to_expiry",
        "scaled_black_price",
        "sabr_rho",
        "sabr_volvol",
    ]
    y_col = "scaled_market_price"
    k_col = "strike_price"

    X_calls = df_calls.select(feature_cols)
    X_puts = df_puts.select(feature_cols)

    y_calls = df_calls.select(y_col)
    y_puts = df_puts.select(y_col)

    K_calls = df_calls.select(k_col)
    K_puts = df_puts.select(k_col)

    X_calls_tmp, X_calls_test, y_calls_tmp, y_calls_test = train_test_split(
        X_calls, y_calls, test_size=0.2, shuffle=False
    )
    X_puts_tmp, X_puts_test, y_puts_tmp, y_puts_test = train_test_split(
        X_puts, y_puts, test_size=0.2, shuffle=False
    )

    split_idx = int(len(K_calls) * 0.8)
    K_calls_tmp, K_calls_test = (
        K_calls.head(split_idx),
        K_calls.tail(len(K_calls) - split_idx),
    )
    split_idx = int(len(K_puts) * 0.8)
    K_puts_tmp, K_puts_test = (
        K_puts.head(split_idx),
        K_puts.tail(len(K_puts) - split_idx),
    )

    # 4. (paper) Train validation test split chronologically: 75% of train = train, 25% of train = validation
    X_calls_train, X_calls_val, y_calls_train, y_calls_val = train_test_split(
        X_calls_tmp, y_calls_tmp, test_size=0.25, shuffle=False
    )
    X_puts_train, X_puts_val, y_puts_train, y_puts_val = train_test_split(
        X_puts_tmp, y_puts_tmp, test_size=0.25, shuffle=False
    )

    split_idx = int(len(K_calls_tmp) * 0.75)
    K_calls_train, K_calls_val = (
        K_calls_tmp.head(split_idx),
        K_calls_tmp.tail(len(K_calls_tmp) - split_idx),
    )
    split_idx = int(len(K_puts_tmp) * 0.75)
    K_puts_train, K_puts_val = (
        K_puts_tmp.head(split_idx),
        K_puts_tmp.tail(len(K_puts_tmp) - split_idx),
    )

    # 5. Convert to torch
    X_calls_train, X_calls_val, X_calls_test = (
        X_calls_train.to_torch(return_type="tensor", dtype=pl.Float32),
        X_calls_val.to_torch(return_type="tensor", dtype=pl.Float32),
        X_calls_test.to_torch(return_type="tensor", dtype=pl.Float32),
    )
    X_puts_train, X_puts_val, X_puts_test = (
        X_puts_train.to_torch(return_type="tensor", dtype=pl.Float32),
        X_puts_val.to_torch(return_type="tensor", dtype=pl.Float32),
        X_puts_test.to_torch(return_type="tensor", dtype=pl.Float32),
    )

    y_calls_train, y_calls_val, y_calls_test = (
        y_calls_train.to_torch(return_type="tensor", dtype=pl.Float32),
        y_calls_val.to_torch(return_type="tensor", dtype=pl.Float32),
        y_calls_test.to_torch(return_type="tensor", dtype=pl.Float32),
    )
    y_puts_train, y_puts_val, y_puts_test = (
        y_puts_train.to_torch(return_type="tensor", dtype=pl.Float32),
        y_puts_val.to_torch(return_type="tensor", dtype=pl.Float32),
        y_puts_test.to_torch(return_type="tensor", dtype=pl.Float32),
    )

    K_calls_train, K_calls_val, K_calls_test = (
        K_calls_train.to_torch(return_type="tensor", dtype=pl.Float32),
        K_calls_val.to_torch(return_type="tensor", dtype=pl.Float32),
        K_calls_test.to_torch(return_type="tensor", dtype=pl.Float32),
    )
    K_puts_train, K_puts_val, K_puts_test = (
        K_puts_train.to_torch(return_type="tensor", dtype=pl.Float32),
        K_puts_val.to_torch(return_type="tensor", dtype=pl.Float32),
        K_puts_test.to_torch(return_type="tensor", dtype=pl.Float32),
    )

    X_calls_train_val, y_calls_train_val, K_calls_train_val = (
        X_calls_tmp.to_torch(return_type="tensor", dtype=pl.Float32),
        y_calls_tmp.to_torch(return_type="tensor", dtype=pl.Float32),
        K_calls_tmp.to_torch(return_type="tensor", dtype=pl.Float32),
    )
    X_puts_train_val, y_puts_train_val, K_puts_train_val = (
        X_puts_tmp.to_torch(return_type="tensor", dtype=pl.Float32),
        y_puts_tmp.to_torch(return_type="tensor", dtype=pl.Float32),
        K_puts_tmp.to_torch(return_type="tensor", dtype=pl.Float32),
    )

    train_calls_dataset = TensorDataset(X_calls_train, y_calls_train, K_calls_train)
    val_calls_dataset = TensorDataset(X_calls_val, y_calls_val, K_calls_val)
    test_calls_dataset = TensorDataset(X_calls_test, y_calls_test, K_calls_test)
    train_val_calls_dataset = TensorDataset(
        X_calls_train_val, y_calls_train_val, K_calls_train_val
    )

    train_puts_dataset = TensorDataset(X_puts_train, y_puts_train, K_puts_train)
    val_puts_dataset = TensorDataset(X_puts_val, y_puts_val, K_puts_val)
    test_puts_dataset = TensorDataset(X_puts_test, y_puts_test, K_puts_test)
    train_val_puts_dataset = TensorDataset(
        X_puts_train_val, y_puts_train_val, K_puts_train_val
    )

    train_calls_loader = DataLoader(
        train_calls_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    val_calls_loader = DataLoader(val_calls_dataset, batch_size=BATCH_SIZE)
    test_calls_loader = DataLoader(test_calls_dataset, batch_size=BATCH_SIZE)

    train_puts_loader = DataLoader(
        train_puts_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    val_puts_loader = DataLoader(val_puts_dataset, batch_size=BATCH_SIZE)
    test_puts_loader = DataLoader(test_puts_dataset, batch_size=BATCH_SIZE)

    # Logging
    ui_elems1 = [
        mo.md("**Input DF:**"),
        df,
        mo.md("**Calls DF:**"),
        df_calls,
        mo.md("**Puts DF:**"),
        df_puts,
        mo.md(f"**Feature cols**: {feature_cols}"),
        mo.md(f"**Target col**: {y_col}"),
        mo.md(f"**Strike col**: {k_col}"),
    ]

    mo.vstack(ui_elems1, align="stretch")

    return (
        train_calls_loader,
        val_calls_loader,
        test_calls_loader,
        train_calls_dataset,
        val_calls_dataset,
        test_calls_dataset,
        train_val_calls_dataset,
        train_puts_loader,
        val_puts_loader,
        test_puts_loader,
        train_puts_dataset,
        val_puts_dataset,
        test_puts_dataset,
        train_val_puts_dataset,
    )


@app.cell(hide_code=True)
def define_model_md():
    mo.md(r"""
    # Define Hybird Model Version 1
    """)


@app.class_definition
class HybridModelV1_SABR(nn.Module):
    def __init__(self, n_layers, n_units, dropout_rate):
        super().__init__()

        layers = []
        input_dim = 7

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


@app.cell(hide_code=True)
def define_train_fn_md():
    mo.md(r"""
    # Define Training Script
    """)


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
        / f"{training_specs['start_date']}_{training_specs['end_date']}_{training_specs['output_filename']}_{training_specs['option_type']}_weights.pt",
    )

    loss_df = pl.DataFrame({"train": train_loss_history, "val": val_loss_history})

    loss_df.write_parquet(
        training_specs["data_dir"]
        / f"{training_specs['start_date']}_{training_specs['end_date']}_{training_specs['output_filename']}_{training_specs['option_type']}_loss_history.parquet"
    )


@app.cell(hide_code=True)
def optuna_objective_md():
    mo.md(r"""
    # Define Optuna Objective
    """)


@app.cell
def optuna_objective(
    train_calls_loader, val_calls_loader, train_puts_loader, val_puts_loader
):
    def objective_calls(trial):
        print("========================================================")
        print(f"Executing Trial {trial.number}")

        # Optuna suggest values
        n_layers = trial.suggest_int("n_layers", 1, 6)
        n_units = trial.suggest_int("n_units", 20, 100)
        dropout_rate = trial.suggest_float("dropout_rate", 0.01, 0.2)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

        # Model specification
        model = HybridModelV1_SABR(n_layers, n_units, dropout_rate).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.L1Loss()

        # Model optimisation
        epoch_tune = 20
        for epoch in range(epoch_tune):
            model.train()

            for batch_X, batch_y, _ in train_calls_loader:
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
                for batch_X, batch_y, _ in val_calls_loader:
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

    def objective_puts(trial):
        print("========================================================")
        print(f"Executing Trial {trial.number}")

        # Optuna suggest values
        n_layers = trial.suggest_int("n_layers", 1, 6)
        n_units = trial.suggest_int("n_units", 20, 100)
        dropout_rate = trial.suggest_float("dropout_rate", 0.01, 0.2)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

        # Model specification
        model = HybridModelV1_SABR(n_layers, n_units, dropout_rate).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.L1Loss()

        # Model optimisation
        epoch_tune = 20
        for epoch in range(epoch_tune):
            model.train()

            for batch_X, batch_y, _ in train_puts_loader:
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
                for batch_X, batch_y, _ in val_puts_loader:
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

    return objective_calls, objective_puts


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
def tune_hyper_calls(objective_calls):
    study_calls = optuna.create_study(
        direction="minimize", study_name="option_nn_calls"
    )

    study_calls.optimize(objective_calls, n_trials=50, n_jobs=-1)

    return study_calls


@app.cell
def tune_hyper_puts(objective_puts):
    study_puts = optuna.create_study(direction="minimize", study_name="option_nn_puts")

    study_puts.optimize(objective_puts, n_trials=50, n_jobs=-1)

    return study_puts


@app.cell(hide_code=True)
def tune_hyper_res_md():
    mo.md(r"""
    # Hyperparameter Optimisation Results
    """)


@app.cell
def tune_hyper_res_calls(study_calls):
    mo.vstack(
        [
            mo.md(f"Best params: {study_calls.best_params}"),
            mo.md(f"Best avg val loss: {study_calls.best_value}"),
            optuna.visualization.plot_optimization_history(study_calls),
            optuna.visualization.plot_timeline(study_calls),
            optuna.visualization.plot_param_importances(study_calls),
        ],
        align="stretch",
    )


@app.cell
def tune_hyper_res_puts(study_puts):
    mo.vstack(
        [
            mo.md(f"Best params: {study_puts.best_params}"),
            mo.md(f"Best avg val loss: {study_puts.best_value}"),
            optuna.visualization.plot_optimization_history(study_puts),
            optuna.visualization.plot_timeline(study_puts),
            optuna.visualization.plot_param_importances(study_puts),
        ],
        align="stretch",
    )


@app.cell(hide_code=True)
def export_md():
    mo.md(r"""
    # Export study
    """)


@app.cell
def export(
    study_calls,
    study_puts,
    train_calls_dataset,
    val_calls_dataset,
    test_calls_dataset,
    train_val_calls_dataset,
    train_puts_dataset,
    val_puts_dataset,
    test_puts_dataset,
    train_val_puts_dataset,
):
    joblib.dump(study_calls, DATA_DIR / f"{START_DATE}_{END_DATE}_study_sabr_calls.pkl")
    joblib.dump(study_puts, DATA_DIR / f"{START_DATE}_{END_DATE}_study_sabr_puts.pkl")

    torch.save(
        train_calls_dataset, DATA_DIR / f"{START_DATE}_{END_DATE}_train_sabr_calls.pt"
    )
    torch.save(
        train_puts_dataset, DATA_DIR / f"{START_DATE}_{END_DATE}_train_sabr_puts.pt"
    )

    torch.save(
        val_calls_dataset, DATA_DIR / f"{START_DATE}_{END_DATE}_val_sabr_calls.pt"
    )
    torch.save(val_puts_dataset, DATA_DIR / f"{START_DATE}_{END_DATE}_val_sabr_puts.pt")

    torch.save(
        test_calls_dataset, DATA_DIR / f"{START_DATE}_{END_DATE}_test_sabr_calls.pt"
    )
    torch.save(
        test_puts_dataset, DATA_DIR / f"{START_DATE}_{END_DATE}_test_sabr_puts.pt"
    )

    torch.save(
        train_val_calls_dataset,
        DATA_DIR / f"{START_DATE}_{END_DATE}_train_val_sabr_calls.pt",
    )
    torch.save(
        train_val_puts_dataset,
        DATA_DIR / f"{START_DATE}_{END_DATE}_train_val_sabr_puts.pt",
    )


if __name__ == "__main__":
    app.run()
