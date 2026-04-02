import os
import sys

import pandas as pd
import pymongo
import certifi
from dotenv import load_dotenv, find_dotenv

from src.exception.exception import RiskyException


class InsertRawDataToMongo:
    def __init__(self, mongo_uri, mongo_database, mongo_collection, dataset_file_path=None, records=None):
        self.mongo_uri = mongo_uri
        self.mongo_database = mongo_database
        self.mongo_collection = mongo_collection
        self.dataset_file_path = dataset_file_path
        self.records = records
        self.ca = certifi.where()


    def check_connection(self):
        print(self.mongo_uri)
        client = pymongo.MongoClient(
            self.mongo_uri,
            tlsCAFile=ca,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
            socketTimeoutMS=20000,
        )
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
        client.close()

    def json_cvs_convertor(self):
        client = pymongo.MongoClient(
            self.mongo_uri,
            tlsCAFile=self.ca,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
            socketTimeoutMS=20000,
        )
        try:
            db = client[self.mongo_database]
            collection = db[self.mongo_collection]
            documents = list(collection.find({}))
            df = pd.DataFrame(documents)
            return df
        except Exception as e:
            raise RiskyException(e, sys)
        finally:
            client.close()

    def csv_json_convertor(self):
        try:
            data = pd.read_csv(self.dataset_file_path)
            data.reset_index(drop=True, inplace=True)
            self.records = data.to_dict(orient="records")
        except Exception as e:
            raise RiskyException(e, sys)

    def insert_data(self):
        client = pymongo.MongoClient(
            self.mongo_uri,
            tlsCAFile=ca,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
            socketTimeoutMS=20000,
        )
        try:
            db = client[self.mongo_database]
            collection = db[self.mongo_collection]
            collection.insert_many(self.records)
            return len(self.records)
        except Exception as e:
            raise RiskyException(e, sys)
        finally:
            client.close()




if __name__ == "__main__":
    load_dotenv(find_dotenv())

    mongo_uri = os.getenv('MONGO_DB_URL')
    dataset_file_path = os.getenv('DATASET_FILE_PATH')
    mongo_database = os.getenv('DATABASE')
    mongo_collection = os.getenv('COLLECTION')
    mongo = InsertRawDataToMongo(mongo_uri, mongo_database, mongo_collection, dataset_file_path)
    mongo.check_connection()
    mongo.csv_json_convertor()
    print(mongo.insert_data())
    print("Mongo inserted successfully")
