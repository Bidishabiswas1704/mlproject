import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exceptions import CustomException
from src.logger import logging
import os

from src.utlis import save_object
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self)
        self.data_transformation_config=DataTransformationConfig

    def get_data_transformer_object(self):
        '''
        This method is used to get the preprocessor object or responsible for data transformation
        '''
        try:
            numerical_columns=[""]
            categorical_columns=[""]

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(statergy="median")),
                ("scaler",StandardScaler())

                ]
            )    
            cat_pipeline=Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("onehot",OneHotEncoder(handle_unknown="ignore")),
                ("scaler",StandardScaler())
                ]
            )
            logging.info("Numerical columns standard scaling completed")

            logging.info("Categorical column encoding completed")
            
            preprocessor=ColumnTransformer(
                transformers=[
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("obaining preprocessing object")
            preprocessor_obj=self.get_data_transformer_object()

            target_column_name=""
            numerical_columns=[""]
        
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying processing object on training dataframe and testing dataframe."

            )
            input_feature_train_arr=preprocessor_obj.fit_tarnsform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.tarnsform(target_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
           



