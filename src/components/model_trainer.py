import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

# MLflow imports
import mlflow
import mlflow.sklearn
import mlflow.catboost
import mlflow.xgboost

from src.logger import logging
from src.exception import OptiPriceException
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def start_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing arrays")

            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(),
                "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
                "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
                "CatBoost": CatBoostRegressor(verbose=0, random_state=42)
            }

            logging.info("Training and evaluation started for all regression models")
            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models
            )

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.60:
                raise Exception("No Model performed better than 60%")

            logging.info(f"Best Model: [{best_model_name}] with R2 Score: [{best_model_score:.5f}]")

            # Local disk par model pickle save karna
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Test set predictions & metric calculation
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            rmse = np.sqrt(mean_squared_error(y_test, predicted))
            mae = mean_absolute_error(y_test, predicted)

            # ---------------- MLflow Tracking Block ----------------
            mlflow.set_experiment("OptiPrice_Pricing_Optimization")

            with mlflow.start_run(run_name=f"Winning_Model_{best_model_name}"):
                # 1. Log parameters
                mlflow.log_param("best_model_name", best_model_name)
                mlflow.log_param("train_samples", len(X_train))
                mlflow.log_param("test_samples", len(X_test))

                # 2. Log evaluation metrics
                mlflow.log_metric("r2_score", r2_square)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)

                # 3. Dedicated MLflow Flavor logging (Skops error fix)
                if best_model_name == "CatBoost":
                    mlflow.catboost.log_model(cb_model=best_model, artifact_path="model")
                elif best_model_name == "XGBoost":
                    mlflow.xgboost.log_model(xgb_model=best_model, artifact_path="model")
                else:
                    mlflow.sklearn.log_model(sk_model=best_model, artifact_path="model")

                logging.info("MLflow run tracking successfully logged")
            # -------------------------------------------------------

            print(f"\n==========================================")
            print(f" Best Model: {best_model_name}")
            print(f" R2 Score:   {r2_square:.5f}")
            print(f" RMSE:       ${rmse:.2f}")
            print(f" MAE:        ${mae:.2f}")
            print(f"==========================================\n")

            return r2_square

        except Exception as e:
            raise OptiPriceException(e, sys)

# if __name__ == "__main__":
#     from src.components.data_ingestion import DataIngestion
#     from src.components.data_transformation import DataTransformation

#     ingestion = DataIngestion()
#     train_data, test_data = ingestion.start_data_ingestion()

#     transformer = DataTransformation()
#     train_arr, test_arr, _ = transformer.start_data_transformation(train_data, test_data)

#     trainer = ModelTrainer()
#     score = trainer.start_model_trainer(train_arr, test_arr)
#     print("Model Training Successfully Completed! Final R2 Score:", score)