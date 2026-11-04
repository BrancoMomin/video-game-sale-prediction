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
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

    def get_data_transformer(self, numeric_cols, categorical_cols):

        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numeric_cols),
                ("cat", categorical_pipeline, categorical_cols),
            ]
        )

        return preprocessor

    def initiate_data_transformation(self, train_path, test_path):

        try:
            logging.info("Starting data transformation")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            if train_df.empty or test_df.empty:
                raise ValueError("Training or testing dataset is empty")

            target_col = "Global_Sales"

            if target_col not in train_df.columns:
                raise KeyError(f"Target column {target_col} not found")

            if target_col not in test_df.columns:
                raise KeyError(f"Target column {target_col} not found")

            if train_df[target_col].isna().to_numpy(dtype=bool).any():
                raise ValueError("Training target contains missing values")

            if test_df[target_col].isna().to_numpy(dtype=bool).any():
                raise ValueError("Testing target contains missing values")

            if not pd.api.types.is_numeric_dtype(train_df[target_col]):
                raise TypeError("Target column must be numeric")

            logging.info("Separating features and target")

            drop_cols = [
                "Global_Sales",
                "Name",
                "Rank",
                "NA_Sales",
                "EU_Sales",
                "JP_Sales",
                "Other_Sales",
            ]

            X_train = train_df.drop(columns=drop_cols, errors="ignore")

            y_train = train_df[target_col]

            X_test = test_df.drop(columns=drop_cols, errors="ignore")

            y_test = test_df[target_col]

            if X_train.empty:
                raise ValueError("No feature columns available")

            X_test = X_test.reindex(columns=X_train.columns)

            numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()

            categorical_cols = X_train.select_dtypes(
                include=["object", "category", "string", "bool"]
            ).columns.tolist()

            logging.info(f"Numeric columns: {numeric_cols}")
            logging.info(f"Categorical columns: {categorical_cols}")

            if not numeric_cols and not categorical_cols:
                raise ValueError("No valid feature columns found")

            logging.info("Creating preprocessing pipeline")

            preprocessor = self.get_data_transformer(numeric_cols, categorical_cols)

            logging.info("Fitting preprocessing pipeline on training data")

            X_train_arr = preprocessor.fit_transform(X_train)

            logging.info("Transforming testing data")

            X_test_arr = preprocessor.transform(X_test)

            os.makedirs("artifacts", exist_ok=True)

            with open(self.preprocessor_path, "wb") as file:
                pickle.dump(preprocessor, file)

            logging.info(
                "Preprocessor saved successfully to artifacts/preprocessor.pkl"
            )

            logging.info("Data transformation completed successfully")

            return X_train_arr, y_train, X_test_arr, y_test

        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            pd.errors.ParserError,
            pd.errors.EmptyDataError,
            pickle.PicklingError,
        ) as e:
            logging.exception("Error occurred during data transformation")

            raise CustomException(e, sys) from e
