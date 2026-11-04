import os
import sys
from typing import cast

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging


class DataIngestion:
    def __init__(self):
        self.raw_data_path = os.path.join("artifacts", "raw.csv")
        self.train_data_path = os.path.join("artifacts", "train.csv")
        self.test_data_path = os.path.join("artifacts", "test.csv")

    def initiate_data_ingestion(self):
        try:
            logging.info("Starting data ingestion")

            df: pd.DataFrame = pd.read_csv("data/raw/vgsales.csv")

            if df.empty:
                raise ValueError("Dataset is empty")

            logging.info(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")

            os.makedirs("artifacts", exist_ok=True)

            df.to_csv(self.raw_data_path, index=False)

            logging.info("Splitting data into train and test sets")

            split_data = train_test_split(df, test_size=0.2, random_state=42)

            train_set, test_set = cast(
                tuple[pd.DataFrame, pd.DataFrame], tuple(split_data)
            )

            train_set.to_csv(self.train_data_path, index=False)
            test_set.to_csv(self.test_data_path, index=False)

            logging.info("Data ingestion completed successfully")

            return self.train_data_path, self.test_data_path

        except (
            OSError,
            ValueError,
            pd.errors.ParserError,
            pd.errors.EmptyDataError,
        ) as e:
            logging.exception("Error occurred during data ingestion")
            raise CustomException(e, sys) from e
