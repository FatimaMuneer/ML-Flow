from pathlib import Path
import warnings
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Suppress harmless stage deprecation warnings in newer MLflow versions
warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow")

# ─── 0. Fix Windows Path Space Issue ─────────────────────
base_dir = Path(__file__).parent.resolve()
db_path = (base_dir / "mlflow.db").as_posix()
tracking_uri = f"sqlite:///{db_path}"

mlflow.set_tracking_uri(tracking_uri)
client = MlflowClient(tracking_uri=tracking_uri)

# ─── 1. Register the model from your best run ───────────
# If logged via autolog, the folder is usually 'model'; if manual, 'ridge_model'
run_id = "3e2bff4a38fa4bcbaebcf9e3769485e9"
model_name = "CaliforniaHousingPredictor"

try:
    model_uri = f"runs:/{run_id}/ridge_model"
    registered = mlflow.register_model(model_uri=model_uri, name=model_name)
except Exception:
    model_uri = f"runs:/{run_id}/model"
    registered = mlflow.register_model(model_uri=model_uri, name=model_name)

print(f"Registered: {registered.name}, version {registered.version}")

# ─── 2. Transition to Staging ───────────────────────────
client.transition_model_version_stage(
    name=model_name,
    version=registered.version,
    stage="Staging",
)
print("Moved to Staging ✓")

# ─── 3. Add a description ───────────────────────────────
client.update_model_version(
    name=model_name,
    version=registered.version,
    description=(
        "Ridge regression, alpha=0.1, trained on California Housing. "
        "Val RMSE=0.7278, Val R²=0.6004."
    ),
)

# ─── 4. Promote to Production after validation ──────────
client.transition_model_version_stage(
    name=model_name,
    version=registered.version,
    stage="Production",
    archive_existing_versions=True,  # auto-archive old prod
)
print("Promoted to Production ✓")

# ─── 5. Load from Production (how downstream services do it)
prod_model = mlflow.sklearn.load_model(
    model_uri=f"models:/{model_name}/Production"
)
print(f"Loaded production model: {prod_model}")