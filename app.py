import sys
from flask import Flask, request, render_template

from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.logger import logging
from src.exception import OptiPriceException

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("home.html")

@app.route("/predict", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")
    else:
        try:
            product_category = request.form.get("product_category")
            base_cost = float(request.form.get("base_cost"))
            competitor_price = float(request.form.get("competitor_price"))
            inventory_level = int(request.form.get("inventory_level"))
            demand_score = float(request.form.get("demand_score"))
            customer_rating = float(request.form.get("customer_rating"))
            is_weekend = int(request.form.get("is_weekend", 0))
            shipping_days = int(request.form.get("shipping_days", 3))

            data = CustomData(
                product_category=product_category,
                base_cost=base_cost,
                competitor_price=competitor_price,
                inventory_level=inventory_level,
                demand_score=demand_score,
                customer_rating=customer_rating,
                is_weekend=is_weekend,
                shipping_days=shipping_days
            )

            pred_df = data.get_data_as_data_frame()
            logging.info(f"Input features received: \n{pred_df}")

            predict_pipeline = PredictPipeline()
            predicted_price = predict_pipeline.predict(pred_df)[0]
            
            optimal_price = round(predicted_price, 2)
            margin = round(optimal_price - base_cost, 2)
            markup_pct = round((margin / base_cost) * 100, 1)

            insight = f"Based on category '{product_category}' and demand score {demand_score}, a {markup_pct}% markup over base cost is recommended."

            return render_template(
                "home.html",
                optimal_price=f"${optimal_price:.2f}",
                base_cost=f"${base_cost:.2f}",
                margin=f"+${margin:.2f}",
                insight=insight,
                prev_category=product_category,
                prev_base_cost=base_cost,
                prev_comp_price=competitor_price,
                prev_inventory=inventory_level,
                prev_demand=demand_score,
                prev_rating=customer_rating,
                prev_weekend=is_weekend,
                prev_shipping=shipping_days
            )

        except Exception as e:
            logging.error(f"Error during live prediction: {str(e)}")
            raise OptiPriceException(e, sys)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)