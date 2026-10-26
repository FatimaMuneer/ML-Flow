from pathlib import Path
import mlflow
import mlflow.sklearn

# Fix Windows space / %20 path issue by explicitly setting a local SQLite tracking URI
db_path = Path(__file__).parent.resolve() / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{db_path}")

# 1. Name your experiment (creates it if it doesn't exist)
mlflow.set_experiment("hello_mlflow_demo")

# 2. Start a run — everything inside this block is recorded
with mlflow.start_run():
    # 3. Log parameters (inputs)
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)

    # 4. Log metrics (outputs you care about)
    mlflow.log_metric("accuracy", 0.92)
    mlflow.log_metric("loss", 0.08)

    # 5. Log tags (freeform metadata)
    mlflow.set_tag("team", "ml-research")
    mlflow.set_tag("dataset_version", "v2.1")

    # 6. The run auto-closes when the with block exits
    print("Run completed successfully!")