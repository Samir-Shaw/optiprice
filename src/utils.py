# import os
# import sys
# import dill
# from sklearn.metrics import r2_score
# from src.exception import OptiPriceException
# from src.logger import logging

# def save_object(file_path, obj):
#     try:
#         dir_path = os.path.dirname(file_path)
#         os.makedirs(dir_path, exist_ok=True)

#         with open(file_path, "wb") as file_obj:
#             dill.dump(obj, file_obj)

#         logging.info(f"Object successfully saved at: {file_path}")

#     except Exception as e:
#         raise OptiPriceException(e, sys)


# def load_object(file_path):
#     try:
#         if not os.path.exists(file_path):
#             raise FileNotFoundError(f"File nahi mili is path par: {file_path}")

#         with open(file_path, "rb") as file_obj:
#             obj = dill.load(file_obj)

#         return obj

#     except Exception as e:
#         raise OptiPriceException(e, sys)


# def evaluate_models(X_train, y_train, X_test, y_test, models):
#     try:
#         report = {}

#         for model_name, model in models.items():
#             model.fit(X_train, y_train)

#             y_test_pred = model.predict(X_test)

#             test_model_score = r2_score(y_test, y_test_pred)

#             report[model_name] = test_model_score
#             logging.info(f"{model_name} training complete | R2 Score: {test_model_score:.4f}")

#         return report

#     except Exception as e:
#         raise OptiPriceException(e, sys)