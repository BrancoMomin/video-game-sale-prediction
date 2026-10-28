import os
import pickle
import sys

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging


class DataTransformation:
    def __init__(self):
        self.preprocessor_path = os.path.join(
            "artifacts", "preprocessor.pkl"
        )

    def get_data_transformer(self, numeric_cols, categorical_cols):
        try:
            numeric_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent"),
                    ),
                    (
                        "onehot",
                        OneHotEncoder(handle_unknown="ignore"),
                    ),
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_pipeline, numeric_cols),
                    ("cat", categorical_pipeline, categorical_cols),
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            target_col = "Global_Sales"
            drop_cols = [target_col, "Name"]

            X_train = train_df.drop(columns=drop_cols)
            y_train = train_df[target_col]

            X_test = test_df.drop(columns=drop_cols)
            y_test = test_df[target_col]

            numeric_cols = X_train.select_dtypes(
                include=["int64", "float64"]
            ).columns.tolist()

            categorical_cols = X_train.select_dtypes(
                include=["object", "str"]
            ).columns.tolist()

            logging.info(f"Numeric columns: {numeric_cols}")
            logging.info(f"Categorical columns: {categorical_cols}")

            preprocessor = self.get_data_transformer(
                numeric_cols, categorical_cols
            )

            X_train_arr = preprocessor.fit_transform(X_train)
            X_test_arr = preprocessor.transform(X_test)

            os.makedirs("artifacts", exist_ok=True)

            with open(self.preprocessor_path, "wb") as file:
                pickle.dump(preprocessor, file)

            logging.info(
                "Saved preprocessing object to artifacts/preprocessor.pkl"
            )

            return X_train_arr, y_train, X_test_arr, y_test

        except Exception as e:
            raise CustomException(e, sys)
