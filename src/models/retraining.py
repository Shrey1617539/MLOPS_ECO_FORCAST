import os
import shutil
import mlflow
from itertools import product
from pathlib import Path
import subprocess
from datetime import datetime
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.logging import get_logger, log_function_call
from src.config import get_cities
from src.models.train import train_model_for_city

logger = get_logger(__name__)

# # Configure MLflow

class ModelRetrainer:
    """Handles model retraining when drift is detected, with MLflow hyperparameter tuning."""
    
    def __init__(self):
        mlflow.set_experiment("weather_forecasting_retraining")
        self.DATA_ROOT = Path(os.getenv("DATA_ROOT", "data"))
        self.MODEL_ROOT = Path(os.getenv("MODEL_ROOT", "models"))
        self.processed_data_dir = self.DATA_ROOT / "processed"
        self.models_dir = self.MODEL_ROOT / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
    @log_function_call()
    def retrain_models_if_needed(self, drift_results):
        
        results = {}
        cities_to_retrain = [
            city for city, r in drift_results.get("results", {}).items()
            if r.get("status")=="drift_detected"
        ]
        if not cities_to_retrain:
            logger.info("No drift detected, skipping retraining")
            return {"status": "no_retraining_needed"}

        logger.info(f"Drift detected for: {', '.join(cities_to_retrain)}")
        # define hyperparameter grid
        batch_sizes = [16, 32]
        learning_rates = [1e-3, 1e-4]
        epochs = 100  # small number for tuning
        
        for city in cities_to_retrain:
            try:
                city_info = next(
                    (c for c in get_cities()
                     if c['name'].lower().replace(" ","_")==city),
                    None
                )
                if not city_info:
                    raise ValueError("City info not found")
                best_metric, best_params, best_model = -float("inf"), None, None
                # grid search
                for bs, lr in product(batch_sizes, learning_rates):
                    with mlflow.start_run(
                        run_name=f"{city}_bs{bs}_lr{lr}"
                    ):
                        mlflow.log_param("batch_size", bs)
                        mlflow.log_param("learning_rate", lr)
                        mlflow.log_param("epochs", epochs)
                        
                        # assumes train_model_for_city returns dict with 'metrics' and 'model_path'
                        train_res = train_model_for_city(
                            city_info,
                            batch_size=bs,
                            learning_rate=lr,
                            epochs=epochs
                        )
                        metrics = train_res.get("metrics", {})
                        # pick validation accuracy or fallback to accuracy
                        metric = metrics.get("val_accuracy", metrics.get("accuracy", 0))
                        for k,v in metrics.items():
                            mlflow.log_metric(k, v)
                        model_path = Path(train_res["model_path"])
                        mlflow.log_artifact(str(model_path))
                        
                        if metric > best_metric:
                            best_metric, best_params, best_model = metric, {"batch_size":bs,"learning_rate":lr}, model_path
                        
                # copy best model to standard location
                dest = self.models_dir / f"{city}_model.h5"
                shutil.copy(best_model, dest)
                # version with DVC
                self._version_model(city)
                
                results[city] = {
                    "status": "retrained",
                    "best_params": best_params,
                    "best_metric": best_metric
                }
                logger.info(f"{city} retrained: params={best_params}, metric={best_metric}")
            
            except Exception as e:
                logger.error(f"Error retraining {city}: {e}")
                results[city] = {"status":"error","error":str(e)}
        
        return {
            "status": "retraining_completed",
            "timestamp": datetime.now().isoformat(),
            "results": results
        }

    def _version_model(self, city_name):
        """Version the retrained model using DVC."""
        try:
            model_file = self.models_dir / f"{city_name}_model.h5"
            subprocess.run(["dvc","add",str(model_file)], check=True)
            subprocess.run(["git","add",f"{model_file}.dvc"], check=True)
            
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            subprocess.run(
                ["git","commit","-m",f"Retrained {city_name} model at {ts}"],
                check=True
            )
            logger.info(f"Versioned {city_name} model with DVC")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"DVC versioning failed: {e}")
            return False
        
if __name__ == "__main__":
    retrainer = ModelRetrainer()
    # Simulate drift detection for the first-time training
    drift_results = {
        "results": {city['name'].lower().replace(" ", "_"): {"status": "drift_detected"} for city in get_cities()}
    }
    retrain_results = retrainer.retrain_models_if_needed(drift_results)
    logger.info(f"Retraining results: {retrain_results}")