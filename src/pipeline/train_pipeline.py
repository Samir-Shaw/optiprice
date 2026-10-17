import sys
from src.logger import logging
from src.exception import OptiPriceException
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            logging.info("Full Training Pipeline Execution Started")

            logging.info("Step 1: Starting Data Ingestion")
            ingestion = DataIngestion()
            train_data_path, test_data_path = ingestion.start_data_ingestion()
            logging.info(f"Data Ingestion Complete. Train: {train_data_path}, Test: {test_data_path}")

            logging.info("Step 2: Starting Data Validation Check")
            validator = DataValidation()
            validation_status = validator.validate_schema(train_data_path)
            
            if not validation_status:
                raise Exception("Data Validation failed! Pipeline stopped")
            logging.info("Data Validation Passed Successfully.")

            logging.info("Step 3: Starting Data Transformation")
            transformer = DataTransformation()
            train_arr, test_arr, preprocessor_path = transformer.start_data_transformation(
                train_path=train_data_path,
                test_path=test_data_path
            )
            logging.info(f"Data Transformation Complete. Preprocessor saved at: {preprocessor_path}")

            logging.info("Step 4: Starting Model Training & Best Model Selection")
            trainer = ModelTrainer()
            r2_score = trainer.start_model_trainer(train_array=train_arr, test_array=test_arr)
            logging.info(f"Model Training Complete. Best Model R2 Score: {r2_score:.5f}")

            logging.info("Full Training Pipeline Execution Finished Successfully")
            return r2_score

        except Exception as e:
            raise OptiPriceException(e, sys)

# if __name__ == "__main__":
#     pipeline = TrainPipeline()
#     final_score = pipeline.run_pipeline()
#     print(f"\nTraining Pipeline Ran Successfully! Winner Model Score: {final_score:.5f}")