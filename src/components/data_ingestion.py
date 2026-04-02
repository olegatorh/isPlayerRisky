import os
import sys

import pandas as pd
from dotenv import load_dotenv, find_dotenv
from src.logging.logger import logging
from src.entity.artifact_entity import DataIngestionArtifact
from src.entity.config_entity import DataIngestionConfig
from src.exception.exception import RiskyException
from src.utils.mongo_utils import InsertRawDataToMongo
from sklearn.model_selection import train_test_split


load_dotenv(find_dotenv())
mongo_url = os.environ.get("MONGO_DB_URL")
mongo_database = os.getenv('DATABASE')
mongo_collection = os.getenv('COLLECTION')


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except RiskyException as e:
            raise RiskyException(e, sys.exc_info())

    def import_data_from_mongo(self):
        try:
            logging.info("Importing data from mongo")
            mongo = InsertRawDataToMongo(mongo_url, mongo_database, mongo_collection)
            df = mongo.json_cvs_convertor()
            feature_store_file_path = self.data_ingestion_config.feature_store_dir
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            df.to_csv(feature_store_file_path, index=False, header=True)
            logging.info("finished importing data from mongo")
            return df
        except Exception as e:
            raise RiskyException(e, sys)

    def split_data_as_train_and_test(self, df: pd.DataFrame):
        try:
            logging.info("Start train test split on the dataframe")
            df = df.drop(columns=["_id"])
            train_df, test_df = train_test_split(
                df, test_size = self.data_ingestion_config.training_pipeline_config.train_test_split_ratio
            )
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)
            logging.info("Exporting train and test file path.")
            train_df.to_csv(
                self.data_ingestion_config.training_file_path, index=False, header=True
            )
            test_df.to_csv(
                self.data_ingestion_config.testing_file_path, index=False, header=True
            )
            logging.info("Exported train and test file path.")
        except Exception as e:
            raise RiskyException(e, sys)


    def initiate_data_ingestion(self):
        try:
            dataframe = self.import_data_from_mongo()
            self.split_data_as_train_and_test(dataframe)
            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )
            return data_ingestion_artifact
        except Exception as e:
            raise RiskyException(e, sys)