from lets_plot import geom_abline
import marimo

app = marimo.App()

with app.setup():
    import copy
    from pathlib import Path

    import joblib
    import lets_plot as lp
    import marimo as mo
    import optuna
    import polars as pl
    import shap
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm
    import matplotlib.pyplot as plt

    from model_prep import HybridModelV1
    from model_prep_sabr import HybridModelV1_SABR

    mo._runtime.context.get_context().marimo_config["runtime"]["output_max_bytes"] = (
        10000000000
    )

    # START_DATE = "2025-07-01"
    # END_DATE = "2025-08-31"
    START_DATE = "2020-09-01"
    END_DATE = "2025-08-29"

    DATA_DIR = Path("data")

    BATCH_SIZE = 512

    DEVICE = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"Using {DEVICE} device")

    # ===============================================
    # Load Processed data
    # ===============================================
    data = pl.read_parquet(
        DATA_DIR / f"{START_DATE}_{END_DATE}_cleaned_data.parquet"
    ).sort(["date", "cp_flag", "exdate", "strike_price"])

    df_calls = data.filter(pl.col("cp_flag") == "C")
    df_puts = data.filter(pl.col("cp_flag") == "P")

    split_idx = int(len(df_calls) * 0.8)
    train_full_calls_data, test_calls_data = (
        df_calls.head(split_idx),
        df_calls.tail(len(df_calls) - split_idx),
    )
    split_idx = int(len(train_full_calls_data) * 0.75)
    train_calls_data, val_calls_data = (
        train_full_calls_data.head(split_idx),
        train_full_calls_data.tail(len(train_full_calls_data) - split_idx),
    )

    split_idx = int(len(df_puts) * 0.8)
    train_full_puts_data, test_puts_data = (
        df_puts.head(split_idx),
        df_puts.tail(len(df_puts) - split_idx),
    )
    split_idx = int(len(train_full_puts_data) * 0.75)
    train_puts_data, val_puts_data = (
        train_full_puts_data.head(split_idx),
        train_full_puts_data.tail(len(train_full_puts_data) - split_idx),
    )

    # ===============================================
    # No SABR features
    # ===============================================
    # Train dataset
    train_calls_dataset = torch.load(
        DATA_DIR / f"{START_DATE}_{END_DATE}_train_calls.pt", weights_only=False
    )
    train_puts_dataset = torch.load(
        DATA_DIR / f"{START_DATE}_{END_DATE}_train_puts.pt", weights_only=False
    )
    train_calls_loader = DataLoader(train_calls_dataset, batch_size=BATCH_SIZE)
    train_puts_loader = DataLoader(train_puts_dataset, batch_size=BATCH_SIZE)

    # Validation dataset
    val_calls_dataset = torch.load(
        DATA_DIR / f"{START_DATE}_{END_DATE}_val_calls.pt", weights_only=False
    )
    val_puts_dataset = torch.load(
        DATA_DIR / f"{START_DATE}_{END_DATE}_val_puts.pt", weights_only=False
    )
    val_calls_loader = DataLoader(val_calls_dataset, batch_size=BATCH_SIZE)
    val_puts_loader = DataLoader(val_puts_dataset, batch_size=BATCH_SIZE)

    # Test dataset
    test_calls_dataset = torch.load(
        DATA_DIR / f"{START_DATE}_{END_DATE}_test_calls.pt", weights_only=False
    )
    test_puts_dataset = torch.load(
        DATA_DIR / f"{START_DATE}_{END_DATE}_test_puts.pt", weights_only=False
    )
    test_calls_loader = DataLoader(test_calls_dataset, batch_size=BATCH_SIZE)
    test_puts_loader = DataLoader(test_puts_dataset, batch_size=BATCH_SIZE)

    # Optuna study
    study_calls = joblib.load(DATA_DIR / f"{START_DATE}_{END_DATE}_study_calls.pkl")
    study_puts = joblib.load(DATA_DIR / f"{START_DATE}_{END_DATE}_study_puts.pkl")

    # ===============================================
    # SABR features
    # ===============================================
    # Train dataset
    train_sabr_calls_dataset = torch.load(
        DATA_DIR / f"{START_DATE}_{END_DATE}_train_sabr_calls.pt", weights_only=False
    )
    train_sabr_puts_dataset = torch.load(
        DATA_DIR / f"{START_DATE}_{END_DATE}_train_sabr_puts.pt", weights_only=False
    )
    train_sabr_calls_loader = DataLoader(
        train_sabr_calls_dataset, batch_size=BATCH_SIZE
    )
    train_sabr_puts_loader = DataLoader(train_sabr_puts_dataset, batch_size=BATCH_SIZE)

    # Validation dataset
    val_sabr_calls_dataset = torch.load(
        DATA_DIR / f"{START_DATE}_{END_DATE}_val_sabr_calls.pt", weights_only=False
    )
    val_sabr_puts_dataset = torch.load(
        DATA_DIR / f"{START_DATE}_{END_DATE}_val_sabr_puts.pt", weights_only=False
    )
    val_sabr_calls_loader = DataLoader(val_sabr_calls_dataset, batch_size=BATCH_SIZE)
    val_sabr_puts_loader = DataLoader(val_sabr_puts_dataset, batch_size=BATCH_SIZE)

    # Test dataset
    test_sabr_calls_dataset = torch.load(
        DATA_DIR / f"{START_DATE}_{END_DATE}_test_sabr_calls.pt", weights_only=False
    )
    test_sabr_puts_dataset = torch.load(
        DATA_DIR / f"{START_DATE}_{END_DATE}_test_sabr_puts.pt", weights_only=False
    )
    test_sabr_calls_loader = DataLoader(test_sabr_calls_dataset, batch_size=BATCH_SIZE)
    test_sabr_puts_loader = DataLoader(test_sabr_puts_dataset, batch_size=BATCH_SIZE)

    # Optuna study
    study_sabr_calls = joblib.load(
        DATA_DIR / f"{START_DATE}_{END_DATE}_study_sabr_calls.pkl"
    )
    study_sabr_puts = joblib.load(
        DATA_DIR / f"{START_DATE}_{END_DATE}_study_sabr_puts.pkl"
    )

    # ===============================================
    # Verify datasets
    # ===============================================
    # No SABR features, Calls
    cnt_train_calls_bool = len(train_calls_data) == len(train_calls_dataset)
    cnt_val_calls_bool = len(val_calls_data) == len(val_calls_dataset)
    cnt_test_calls_bool = len(test_calls_data) == len(test_calls_dataset)

    # No SABR features, Puts
    cnt_train_puts_bool = len(train_puts_data) == len(train_puts_dataset)
    cnt_val_puts_bool = len(val_puts_data) == len(val_puts_dataset)
    cnt_test_puts_bool = len(test_puts_data) == len(test_puts_dataset)

    # With SABR features, Calls
    cnt_train_sabr_calls_bool = len(train_calls_data) == len(train_sabr_calls_dataset)
    cnt_val_sabr_calls_bool = len(val_calls_data) == len(val_sabr_calls_dataset)
    cnt_test_sabr_calls_bool = len(test_calls_data) == len(test_sabr_calls_dataset)

    # With SABR features, Puts
    cnt_train_sabr_puts_bool = len(train_puts_data) == len(train_sabr_puts_dataset)
    cnt_val_sabr_puts_bool = len(val_puts_data) == len(val_sabr_puts_dataset)
    cnt_test_sabr_puts_bool = len(test_puts_data) == len(test_sabr_puts_dataset)

    print(
        f"(No SABR, Calls) Check Count of Train, Validation, Test datasets: ({cnt_train_calls_bool}, {cnt_val_calls_bool}, {cnt_test_calls_bool})"
    )
    print(
        f"(No SABR, Puts) Check Count of Train, Validation, Test datasets: ({cnt_train_puts_bool}, {cnt_val_puts_bool}, {cnt_test_puts_bool})"
    )
    print(
        f"(With SABR, Calls) Check Count of Train, Validation, Test datasets: ({cnt_train_calls_bool}, {cnt_val_calls_bool}, {cnt_test_calls_bool})"
    )
    print(
        f"(With SABR, Puts) Check Count of Train, Validation, Test datasets: ({cnt_train_puts_bool}, {cnt_val_puts_bool}, {cnt_test_puts_bool})"
    )

    # ===============================================
    # NN Models
    # ===============================================
    # Paper model
    # Calls
    paper_model_calls = HybridModelV1(n_layers=2, n_units=40, dropout_rate=0.1)
    paper_model_calls_weights = torch.load(
        DATA_DIR / f"{START_DATE}_{END_DATE}_paper_calls_weights.pt",
        map_location=DEVICE,
    )
    paper_model_calls.load_state_dict(paper_model_calls_weights)
    paper_model_calls.to(DEVICE)

    # Puts
    paper_model_puts = HybridModelV1(n_layers=2, n_units=40, dropout_rate=0.1)
    paper_model_puts_weights = torch.load(
        DATA_DIR / f"{START_DATE}_{END_DATE}_paper_puts_weights.pt",
        map_location=DEVICE,
    )
    paper_model_puts.load_state_dict(paper_model_puts_weights)
    paper_model_puts.to(DEVICE)

    # Own model
    # Calls
    sabr_model_calls = HybridModelV1_SABR(
        n_layers=study_sabr_calls.best_params["n_layers"],
        n_units=study_sabr_calls.best_params["n_units"],
        dropout_rate=study_sabr_calls.best_params["dropout_rate"],
    )
    sabr_model_calls_weights = torch.load(
        DATA_DIR / f"{START_DATE}_{END_DATE}_optuna_sabr_calls_weights.pt",
        map_location=DEVICE,
    )
    sabr_model_calls.load_state_dict(sabr_model_calls_weights)
    sabr_model_calls.to(DEVICE)

    # Puts
    sabr_model_puts = HybridModelV1_SABR(
        n_layers=study_sabr_puts.best_params["n_layers"],
        n_units=study_sabr_puts.best_params["n_units"],
        dropout_rate=study_sabr_puts.best_params["dropout_rate"],
    )
    sabr_model_puts_weights = torch.load(
        DATA_DIR / f"{START_DATE}_{END_DATE}_optuna_sabr_puts_weights.pt",
        map_location=DEVICE,
    )
    sabr_model_puts.load_state_dict(sabr_model_puts_weights)
    sabr_model_puts.to(DEVICE)


@app.cell(hide_code=True)
def test_preview_md():
    breakdown_df = (
        pl.concat([test_calls_data, test_puts_data])
        .group_by(["cp_flag", "op_level"])
        .count()
    ).sort(["cp_flag", "op_level"])

    mo.vstack(
        [
            mo.md("# Preview Test Data"),
            mo.md(
                f"Total test data size = {len(test_calls_data) + len(test_puts_data):,}"
            ),
            mo.md("Breakdown:"),
            breakdown_df,
            mo.md("Calls:"),
            test_calls_data,
            mo.md("Puts:"),
            test_puts_data,
        ]
    )


@app.cell(hide_code=True)
def optuna_normal_viz_md():
    mo.md(r"""# Optuna Visualisation (Normal)""")


@app.cell
def optuna_normal_viz():
    _1 = study_calls.best_params

    _2 = optuna.visualization.plot_optimization_history(study_calls)

    _3 = optuna.visualization.plot_timeline(study_calls)

    _4 = optuna.visualization.plot_param_importances(study_calls)

    _5 = study_puts.best_params

    _6 = optuna.visualization.plot_optimization_history(study_puts)

    _7 = optuna.visualization.plot_timeline(study_puts)

    _8 = optuna.visualization.plot_param_importances(study_puts)

    mo.vstack(
        [mo.md(r"**Calls:**"), _1, _2, _3, _4, mo.md(r"**Puts**"), _5, _6, _7, _8],
        align="stretch",
    )


@app.cell(hide_code=True)
def optuna_sabr_viz_md():
    mo.md(r"""# Optuna Visualisation (SABR)""")


@app.cell
def optuna_sabr_viz():
    _1 = study_sabr_calls.best_params

    _2 = optuna.visualization.plot_optimization_history(study_sabr_calls)

    _3 = optuna.visualization.plot_timeline(study_sabr_calls)

    _4 = optuna.visualization.plot_param_importances(study_sabr_calls)

    _5 = study_sabr_puts.best_params

    _6 = optuna.visualization.plot_optimization_history(study_sabr_puts)

    _7 = optuna.visualization.plot_timeline(study_sabr_puts)

    _8 = optuna.visualization.plot_param_importances(study_sabr_puts)

    mo.vstack(
        [mo.md(r"**Calls:**"), _1, _2, _3, _4, mo.md(r"**Puts**"), _5, _6, _7, _8],
        align="stretch",
    )


@app.cell(hide_code=True)
def test_analysis_1_md():
    mo.md(r"# Test Analysis")


@app.function
def model_predict(model, test_loader, device):
    model.eval()

    all_preds = []
    all_targets = []
    all_strikes = []

    print(f"Starting {model.__class__.__name__} inference on {device}")

    with torch.no_grad():
        for inputs, targets, strikes in tqdm(
            test_loader, desc="Predicting...", leave=False
        ):
            inputs = inputs.to(device)

            outputs = model(inputs)

            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())
            all_strikes.append(strikes.cpu())

    preds_tensor = torch.cat(all_preds).flatten()
    targets_tensor = torch.cat(all_targets).flatten()
    strikes_tensor = torch.cat(all_strikes).flatten()

    print(f"{model.__class__.__name__} completed")

    return (
        preds_tensor.numpy() * strikes_tensor.numpy(),
        targets_tensor.numpy() * strikes_tensor.numpy(),
    )


@app.cell
def test_analysis_1():
    # Process Calls
    y_pred_paper, y_true_paper = model_predict(
        paper_model_calls, test_calls_loader, DEVICE
    )
    y_pred_sabr, y_true_sabr = model_predict(
        sabr_model_calls, test_sabr_calls_loader, DEVICE
    )
    combined_test_calls_data = test_calls_data.with_columns(
        pl.Series("nn_paper_price", y_pred_paper),
        pl.Series("nn_sabr_price", y_pred_sabr),
        pl.Series("actual_paper", y_true_paper),
        pl.Series("actual_sabr", y_true_sabr),
    )

    # Check calls
    _1 = (
        combined_test_calls_data.with_columns(
            diff1=(pl.col("mid_price") - pl.col("actual_paper")).abs(),
            diff2=(pl.col("mid_price") - pl.col("actual_sabr")).abs(),
        )
        .filter((pl.col("diff1") > 0.01) | (pl.col("diff2") > 0.01))
        .select("mid_price", "actual_paper", "diff1", "diff2")
    )

    # Process Puts
    y_pred_paper, y_true_paper = model_predict(
        paper_model_puts, test_puts_loader, DEVICE
    )
    y_pred_sabr, y_true_sabr = model_predict(
        sabr_model_puts, test_sabr_puts_loader, DEVICE
    )
    combined_test_puts_data = test_puts_data.with_columns(
        pl.Series("nn_paper_price", y_pred_paper),
        pl.Series("nn_sabr_price", y_pred_sabr),
        pl.Series("actual_paper", y_true_paper),
        pl.Series("actual_sabr", y_true_sabr),
    )

    # Check puts
    _2 = (
        combined_test_calls_data.with_columns(
            diff1=(pl.col("mid_price") - pl.col("actual_paper")).abs(),
            diff2=(pl.col("mid_price") - pl.col("actual_sabr")).abs(),
        )
        .filter((pl.col("diff1") > 0.01) | (pl.col("diff2") > 0.01))
        .select("mid_price", "actual_paper", "diff1", "diff2")
    )

    # Combined
    results_df = pl.concat([combined_test_calls_data, combined_test_puts_data]).rename(
        {
            "black_price": "Black Model",
            "sabr_price": "SABR Model",
            "nn_paper_price": "Paper's Neural Network",
            "nn_sabr_price": "Our Extended Neural Network",
            "mid_price": "Actual Market Option Price",
        }
    )

    # Output for easier subsequent loading
    results_df.write_parquet(DATA_DIR / f"{START_DATE}_{END_DATE}_test_results.parquet")

    mo.vstack(
        [
            mo.md(r"(Calls) Check for misalignment of rows"),
            _1,
            mo.md(r"(Puts) Check for misalignment of rows"),
            _2,
            mo.md(r"Result DF:"),
            results_df,
        ],
        align="stretch",
    )

    return results_df


@app.cell
def test_analysis_2(results_df):
    results_df_calls_unpivot = results_df.filter(
        (pl.col("date") == pl.datetime(year=2025, month=4, day=3))
        & (pl.col("cp_flag") == "C")
    ).unpivot(
        on=[
            "Black Model",
            "SABR Model",
            "Paper's Neural Network",
            "Our Extended Neural Network",
        ],
        index=["strike_price", "Actual Market Option Price"],
        variable_name="model",
        value_name="Predicted Price",
    )

    compare_models_calls_scatter = (
        lp.ggplot(
            data=results_df_calls_unpivot,
            mapping=lp.aes(
                x="Actual Market Option Price", y="Predicted Price", color="model"
            ),
        )
        + lp.geom_point(size=2, alpha=0.5)
        + lp.geom_abline(slope=1, intercept=0, color="gray", linetype="dashed")
        + lp.facet_wrap(facets="model", ncol=2)
        + lp.ggtitle("Model Comparison for Calls on 2025-04-03")
        + lp.theme(legend_position="none")
        + lp.theme(plot_title=lp.element_text(hjust=0.5, face="bold", size=14))
    )

    results_df_puts_unpivot = results_df.filter(
        (pl.col("date") == pl.datetime(year=2025, month=4, day=3))
        & (pl.col("cp_flag") == "P")
    ).unpivot(
        on=[
            "Black Model",
            "SABR Model",
            "Paper's Neural Network",
            "Our Extended Neural Network",
        ],
        index=["strike_price", "Actual Market Option Price"],
        variable_name="model",
        value_name="Predicted Price",
    )

    compare_models_puts_scatter = (
        lp.ggplot(
            data=results_df_puts_unpivot,
            mapping=lp.aes(
                x="Actual Market Option Price", y="Predicted Price", color="model"
            ),
        )
        + lp.geom_point(size=2, alpha=0.5)
        + lp.geom_abline(slope=1, intercept=0, color="gray", linetype="dashed")
        + lp.facet_wrap(facets="model", ncol=2)
        + lp.ggtitle("Model Comparison for Puts on 2025-04-03")
        + lp.theme(legend_position="none")
        + lp.theme(plot_title=lp.element_text(hjust=0.5, face="bold", size=14))
    )

    model_cols = [
        "Black Model",
        "SABR Model",
        "Paper's Neural Network",
        "Our Extended Neural Network",
    ]
    compare_mae_table = (
        results_df.select(
            [
                (pl.col(model) - pl.col("Actual Market Option Price"))
                .abs()
                .mean()
                .alias(model)
                for model in model_cols
            ]
        ).unpivot(
            on=[
                "Black Model",
                "SABR Model",
                "Paper's Neural Network",
                "Our Extended Neural Network",
            ],
            variable_name="Model",
            value_name="MAE",
        )
    ).sort(by="MAE")

    compare_mae_moneyness_table = (
        results_df.group_by(["op_level", "cp_flag"])
        .agg(
            [
                (pl.col(model) - pl.col("Actual Market Option Price"))
                .abs()
                .mean()
                .alias(f"{model}")
                for model in model_cols
            ]
        )
        .sort(["cp_flag", "op_level"])
    )

    lp.ggsave(compare_models_calls_scatter, "compare_models_calls_scatter.png")
    lp.ggsave(compare_models_puts_scatter, "compare_models_puts_scatter.png")

    mo.vstack(
        [
            mo.md("Model Comparison (Calls):"),
            compare_models_calls_scatter,
            mo.md("Model Comparison (Puts):"),
            compare_models_puts_scatter,
            mo.md("Overall MAE Results:"),
            compare_mae_table,
            mo.md("MAE Results by Moneyness:"),
            compare_mae_moneyness_table,
        ]
    )


@app.cell(hide_code=True)
def shap_md():
    mo.md("# Shap Analysis")


@app.cell
def run_shap_analysis():
    plt.style.use("seaborn-v0_8-whitegrid")
    sabr_model_calls.eval()
    sabr_model_puts.eval()
    paper_model_calls.eval()
    paper_model_puts.eval()

    specs = [
        (
            "Paper's Neural Network (Calls)",
            paper_model_calls,
            train_calls_loader,
            test_calls_loader,
        ),
        (
            "Paper's Neural Network (Puts)",
            paper_model_puts,
            train_puts_loader,
            test_puts_loader,
        ),
        (
            "Our Extended Neural Network (Calls)",
            sabr_model_calls,
            train_sabr_calls_loader,
            test_sabr_calls_loader,
        ),
        (
            "Our Extended Neural Network (Puts)",
            sabr_model_puts,
            train_sabr_puts_loader,
            test_sabr_puts_loader,
        ),
    ]

    base_features = [
        "Moneyness",
        "Risk-Free Rate",
        "Previous Day VIX",
        "Time to Expiry",
        "Black Price Scaled",
    ]
    sabr_features = base_features + ["SABR Rho", "SABR Nu"]
    for name, m, train_loader, test_loader in specs:
        train_batch = next(iter(train_loader))
        test_batch = next(iter(test_loader))

        train_inputs, _, _ = train_batch
        test_inputs, _, _ = test_batch

        background = train_inputs  # has 512
        test_samples = test_inputs[100:200]

        background = background.to(DEVICE)
        test_samples = test_samples.to(DEVICE)

        explainer = shap.DeepExplainer(m, background)

        shap_values = explainer.shap_values(test_samples)[:, :, 0]

        current_features = sabr_features if shap_values.shape[1] == 7 else base_features

        fig = plt.figure(figsize=(10, 5))
        plt.title(f"SHAP Analysis: {name}", weight="bold")

        shap.summary_plot(
            shap_values,
            test_samples.cpu().numpy(),
            feature_names=current_features,
            plot_type="bar",
        )


if __name__ == "__main__":
    app.run()
