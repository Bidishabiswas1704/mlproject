import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    AdaBoostRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegessor

from src.exceptions import CustomException
from src.logger import logging

from src.utlis import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array,preprocessor_path)
        try:
            logging.info("Split training and test input data")
            X_train, X_test, y_train, y_test =(
                train_array[:,:-1],
                train_array[:,-1], 
                test_array[:,:-1],
                test_array[:,-1] 
            )
            models= {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "K-Neighbour Classifier": KNeighborsRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost Classifier": AdaBoostRegressor(),
                "XGBClassifier": XGBRegessor(),
                "Linear Regression": LinearRegression(),
                "CatBoosting Classifier": CatBoostRegressor()
            }
            params= {
                "Random Forest":{
                    # 'criterion' : ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    "n_estimators": [8,16,32,64,128,256],
                    #"max_features": ["auto", "sqrt", "log2","none"],
                },
                "Decision Tree":{
                    # 'criterion' : ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    "max_depth": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 3]
                    #"max_features": ["auto", "sqrt", "log2","none"],
                    # 'splitter' : ['best', 'randon']
                },
                "AdaBoost Classifier":{
                    "n_estimators" : [8,16,32,64,128,256],
                    'learning_rate': [.1,.01,.05,.001],
                    # 'loss' : ['linear', 'square', 'exponential']
                },
                "K-Neighbour Classifier":{
                    "n_neighbors": [5,7,9,11],
                    # 'weights': ['uniform', 'distance'],
                    # 'algorithm': ['ball_tree', 'kd_tree', 'brute']
                },
                "Gradient Boosting":{
                    # 'loss' : ['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate': [.1,.01,.05,.001],
                    # 'criterion' : ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    "n_estimators": [8,16,32,64,128,256],
                    #"max_features": ["auto", "sqrt", "log2","none"]
                },
                "XGBClassifier":{
                    "n_estimators": [8,16,32,64,128,256],
                    "learning_rate": [.1,.01,.05,.001]
                },
                "Linear Regression":{
                    # 'criterion' : ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    "fit_intercept": [True, False]
                    #"max_features": ["auto", "sqrt", "log2","none"],
                },
                "CatBoosting Classifier":{
                   'learning_rate': [.1,.01,.05,.001],
                    'depth' : [6,8,10],
                    'iterations' : [30, 50, 100]
                }
            }
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param=params)

            ##To get best model score from dict 
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            r2_score= r2_score(y_test,predicted)

        except Exception as e:
            raise CustomException(e,sys)
        
        