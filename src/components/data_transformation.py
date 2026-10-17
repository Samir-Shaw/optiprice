import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.logger import logging
from src.exception import OptiPriceException
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            num_features = [
                'base_cost',
                'competitor_price',
                'inventory_level',
                'demand_score',
                'customer_rating',
                'is_weekend',
                'shipping_days'
            ]
            
            cat_features = ['product_category']

            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ])

            logging.info("Numerical aur Categorical pipelines created")

            preprocessor = ColumnTransformer(transformers=[
                ('num_pipe', num_pipeline, num_features),
                ('cat_pipe', cat_pipeline, cat_features)
            ])

            return preprocessor

        except Exception as e:
            raise OptiPriceException(e, sys)

    def start_data_transformation(self, train_path, test_path):
        try:
            validation_status_file = os.path.join("artifacts", "validation_status.txt")
            if os.path.exists(validation_status_file):
                with open(validation_status_file, "r") as f:
                    status_line = f.read().strip()
                if "True" not in status_line:
                    raise Exception("Data Validation status Failed")
            
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and test data fetched successfullly")

            preprocessor_obj = self.get_data_transformer_object()

            target_column = "optimal_price"
            drop_columns = [target_column, "product_id"]

            input_feature_train_df = train_df.drop(columns=drop_columns, errors='ignore')
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=drop_columns, errors='ignore')
            target_feature_test_df = test_df[target_column]

            logging.info("Preprocessor transformation is happening")

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            logging.info("Preprocessor pickle file artifacts me successfully save ho gayi")

            return (
                train_arr,
                test_arr,
                self.transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise OptiPriceException(e, sys)

# if __name__ == "__main__":
#     train_file = os.path.join("artifacts", "train.csv")
#     test_file = os.path.join("artifacts", "test.csv")
    
#     transform_obj = DataTransformation()
#     train_arr, test_arr, pkl_path = transform_obj.start_data_transformation(train_file, test_file)
    
#     print("Transformation Successful!")
#     print("Train Array Shape:", train_arr.shape)
#     print("Test Array Shape:", test_arr.shape)
#     print("Preprocessor Path:", pkl_path)