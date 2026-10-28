import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from src.logger import logging
from src.exception import CustomException


class DataIngestion:
    def __init__(self):
        self.raw_data_path = os.path.join("artifacts", "raw.csv")
        self.train_data_path = os.path.join("artifacts", "train.csv")
        self.test_data_path = os.path.join("artifacts", "test.csv")

    def initiate_data_ingestion(self):
        try:
            logging.info("Starting data ingestion")

            df = pd.read_csv("data/raw/vgsales.csv")

            logging.info(
                f"Loaded {df.shape[0]} rows and {df.shape[1]} columns"
            )

            os.makedirs("artifacts", exist_ok=True)

            df.to_csv(self.raw_data_path, index=False)

            logging.info("Splitting data into train and test sets")

            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42,
            )

            train_set.to_csv(self.train_data_path, index=False)
            test_set.to_csv(self.test_data_path, index=False)

            logging.info("Data ingestion completed successfully")

            return self.train_data_path, self.test_data_path

        except Exception as e:
            raise CustomException(e, sys)
