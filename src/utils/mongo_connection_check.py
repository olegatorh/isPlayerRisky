import os

from pymongo import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv())
uri = os.environ.get("MONGO_DB_URL")
client = MongoClient(uri, server_api=ServerApi('1'))

try:
    print(uri)
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)