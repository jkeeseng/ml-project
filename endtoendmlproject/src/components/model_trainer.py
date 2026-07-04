import os 
import sys 
from dataclasses import dataclass

from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor 

try:
    from catboost import CatBoostRegressor
except Exception as e:
    CatBoostRegressor = None
    CATBOOST_IMPORT_ERROR = e

try:
    from xgboost import XGBRegressor
except Exception as e:
    XGBRegressor = None
    XGBOOST_IMPORT_ERROR = e

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass 

class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            if XGBRegressor is not None:
                models["XGBRegressor"] = XGBRegressor()
            else:
                logging.warning(f"Skipping XGBRegressor: {XGBOOST_IMPORT_ERROR}")

            if CatBoostRegressor is not None:
                models["CatBoosting Regressor"] = CatBoostRegressor(verbose=False)
            else:
                logging.warning(f"Skipping CatBoosting Regressor: {CATBOOST_IMPORT_ERROR}")

            model_report: dict = evaluate_model(X_train= X_train, y_train= y_train, X_test = X_test, y_test = y_test, models = models)

            #To get the best model score from the dictionary
            best_model_score = max(sorted(model_report.values()))

            #To get the best model name from the dictionary
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            
            logging.info("Best found model on both training and testing dataset")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square
        
        except Exception as e:
            raise CustomException(e, sys)

        
