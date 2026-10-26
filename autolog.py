from pathlib import Path
import warnings
import matplotlib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

# Suppress harmless serialization notices
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")

matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
#  0. FIX WINDOWS PATH SPACE ISSUE
# ─────────────────────────────────────────────
base_dir = Path(__file__).parent.resolve()
db_path = (base_dir / "mlflow.db").as_posix()
mlflow.set_tracking_uri(f"sqlite:///{db_path}")

# ─────────────────────────────────────────────
#  1. LOAD AND PREPARE DATA
# ─────────────────────────────────────────────
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Train / validation / test split (70/15/15)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

# ─────────────────────────────────────────────
#  2. DEFINE HYPERPARAMETER GRID
# ─────────────────────────────────────────────
alpha_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

# ─────────────────────────────────────────────
#  3. SET EXPERIMENT (WITH ARTIFACT ROUTING)
# ─────────────────────────────────────────────
exp_name = "ridge_california_housing"
artifacts_dir = (base_dir / "mlruns").as_posix()

experiment = mlflow.get_experiment_by_name(exp_name)
if experiment is None:
    mlflow.create_experiment(
        name=exp_name,
        artifact_location=f"file:///{artifacts_dir}",
    )

mlflow.set_experiment(exp_name)

# Enable autologging once before starting the loop
# autolog captures model params, training metrics, and logs the model automatically
mlflow.sklearn.autolog(log_models=True)

# ─────────────────────────────────────────────
#  4. TRAINING LOOP — one MLflow run per alpha
# ─────────────────────────────────────────────
for alpha in alpha_values:

    run_name = f"ridge_alpha_{alpha}"

    with mlflow.start_run(run_name=run_name):

        # — Train the model (autolog automatically records params, model, and train metrics here)
        model = Ridge(alpha=alpha)
        model.fit(X_train_s, y_train)

        # — Predict on validation set
        y_pred_val = model.predict(X_val_s)

        # — Compute custom validation metrics
        rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
        mae = mean_absolute_error(y_val, y_pred_val)
        r2 = r2_score(y_val, y_pred_val)

        # Explicitly log validation metrics (autolog captures training set metrics by default)
        mlflow.log_metric("val_rmse", round(rmse, 4))
        mlflow.log_metric("val_mae", round(mae, 4))
        mlflow.log_metric("val_r2", round(r2, 4))

        # — Create and log a custom residuals plot as an artifact
        residuals = y_val - y_pred_val
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(y_pred_val, residuals, alpha=0.4, s=8)
        ax.axhline(y=0, color="red", linestyle="--")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residual")
        ax.set_title(f"Residuals — alpha={alpha}")
        plot_path = f"residuals_alpha_{alpha}.png"
        plt.savefig(plot_path, dpi=80, bbox_inches="tight")
        plt.close()

        mlflow.log_artifact(plot_path)

        # Clean console status output
        print(f"alpha={alpha:7} | RMSE={rmse:.4f} | MAE={mae:.4f} | R²={r2:.4f}")

print("\n✅ All runs logged cleanly. Open http://127.0.0.1:5000 to compare.")