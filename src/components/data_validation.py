import os
import sys
import pandas as pd
from dataclasses import dataclass

from src.logger import logging
from src.exception import OptiPriceException

@dataclass
class DataValidationConfig:
    status_file_path: str = os.path.join("artifacts", "validation_status.txt")

class DataValidation:
    def __init__(self):
        self.validation_config = DataValidationConfig()
        
        self.expected_columns = [
            'product_id',
            'product_category',
            'base_cost',
            'competitor_price',
            'inventory_level',
            'demand_score',
            'customer_rating',
            'is_weekend',
            'shipping_days',
            'optimal_price'
        ]

    def validate_schema(self, file_path: str) -> bool:
        try:
            logging.info(f"Data validation checking : {file_path}")
            df = pd.read_csv(file_path)
            current_columns = list(df.columns)

            validation_status = True

            for col in self.expected_columns:
                if col not in current_columns:
                    validation_status = False
                    logging.warning(f"Data Validation Alert: Missing column -> {col}")
                    break

            os.makedirs(os.path.dirname(self.validation_config.status_file_path), exist_ok=True)
            with open(self.validation_config.status_file_path, "w") as f:
                f.write(f"Validation status: {validation_status}")

            if validation_status:
                logging.info("Data Validation Passed: All 10 cols perfectly matched")
            else:
                logging.error("Data Validation Failed: Schema mismatch detected")

            return validation_status

        except Exception as e:
            raise OptiPriceException(e, sys)

# if __name__ == "__main__":
#     validator = DataValidation()
#     test_file = os.path.join("artifacts", "train.csv")
#     is_valid = validator.validate_schema(test_file)
#     print(f"Schema Validation Result: {is_valid}")