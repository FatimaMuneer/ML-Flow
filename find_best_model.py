from pathlib import Path
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ─── 0. Fix Windows Path Space Issue ─────────────────────
base_dir = Path(__file__).parent.resolve()
db_path = (base_dir / "mlflow.db").as_posix()
mlflow.set_tracking_uri(f"sqlite:///{db_path}")

# ─── 1. Search all runs in the experiment ───────────────
experiment_name = "ridge_california_housing"

runs_df = mlflow.search_runs(
    experiment_names=[experiment_name],
    order_by=["metrics.val_rmse ASC"],  # lowest RMSE first
)

if runs_df.empty:
    print(
        f"❌ No runs found in experiment '{experiment_name}'. Run your training script first."
    )
    exit()

print("=== All Runs (sorted by val_rmse) ===")
columns_to_show = [
    col
    for col in [
        "run_id",
        "params.alpha",
        "metrics.val_rmse",
        "metrics.val_mae",
        "metrics.val_r2",
    ]
    if col in runs_df.columns
]
print(runs_df[columns_to_show].to_string(index=False))

# ─── 2. Extract the best run ─────────────────────────────
best_run = runs_df.iloc[0]  # first row = lowest RMSE
best_run_id = best_run["run_id"]
best_alpha = best_run.get("params.alpha", "N/A")
best_rmse = best_run.get("metrics.val_rmse", np.nan)
best_r2 = best_run.get("metrics.val_r2", np.nan)

print(f"\n🏆 Best Run:")
print(f"   Run ID : {best_run_id}")
print(f"   Alpha  : {best_alpha}")
print(f"   RMSE   : {best_rmse:.4f}")
print(f"   R²     : {best_r2:.4f}")

# ─── 3. Load the best model directly from MLflow ─────────
# When logged via autolog, model artifact folder is named 'model'
# When logged via manual mlflow.sklearn.log_model, it is 'ridge_model'
try:
    model_uri = f"runs:/{best_run_id}/ridge_model"
    best_model = mlflow.sklearn.load_model(model_uri)
except Exception:
    model_uri = f"runs:/{best_run_id}/model"
    best_model = mlflow.sklearn.load_model(model_uri)

print(f"\n✅ Model loaded successfully from: {model_uri}")

# ─── 4. Evaluate on the held-out TEST set ────────────────
housing = fetch_california_housing()
X, y = housing.data, housing.target

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)
_, X_test, _, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

# Fit scaler on actual training split to match training pipeline
scaler = StandardScaler().fit(X_train)
X_test_s = scaler.transform(X_test)

y_pred_test = best_model.predict(X_test_s)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_r2 = r2_score(y_test, y_pred_test)

print(f"\n📊 Final Test Set Performance:")
print(f"   Test RMSE : {test_rmse:.4f}")
print(f"   Test R²   : {test_r2:.4f}")