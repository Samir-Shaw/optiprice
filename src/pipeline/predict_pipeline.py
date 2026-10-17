import os
import sys
import pandas as pd

from src.exception import OptiPriceException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

    def predict(self, features: pd.DataFrame):
        try:
            logging.info("Prediction pipeline started")

            preprocessor = load_object(file_path=self.preprocessor_path)
            model = load_object(file_path=self.model_path)

            logging.info("Features scaling and encoding started")
            data_scaled = preprocessor.transform(features)

            logging.info("Model prediction started")
            preds = model.predict(data_scaled)

            return preds

        except Exception as e:
            raise OptiPriceException(e, sys)


class CustomData:
    def __init__(
        self,
        product_category: str,
        base_cost: float,
        competitor_price: float,
        inventory_level: int,
        demand_score: float,
        customer_rating: float,
        is_weekend: int,
        shipping_days: int
    ):
        self.product_category = product_category
        self.base_cost = base_cost
        self.competitor_price = competitor_price
        self.inventory_level = inventory_level
        self.demand_score = demand_score
        self.customer_rating = customer_rating
        self.is_weekend = is_weekend
        self.shipping_days = shipping_days

    def get_data_as_data_frame(self) -> pd.DataFrame:
        try:
            custom_data_input_dict = {
                "product_category": [self.product_category],
                "base_cost": [self.base_cost],
                "competitor_price": [self.competitor_price],
                "inventory_level": [self.inventory_level],
                "demand_score": [self.demand_score],
                "customer_rating": [self.customer_rating],
                "is_weekend": [self.is_weekend],
                "shipping_days": [self.shipping_days],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise OptiPriceException(e, sys)


# if __name__ == "__main__":
#     sample_data = CustomData(
#         product_category="Electronics",
#         base_cost=120.0,
#         competitor_price=155.0,
#         inventory_level=85,
#         demand_score=1.8,
#         customer_rating=4.5,
#         is_weekend=1,
#         shipping_days=2
#     )

#     df_sample = sample_data.get_data_as_data_frame()
#     print("Input Sample DataFrame:\n", df_sample)

#     pipeline = PredictPipeline()
#     result = pipeline.predict(df_sample)
#     print(f"\nRecommended Optimal Price: ${result[0]:.2f}")