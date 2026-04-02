import os
import pickle
import sys

import numpy as np
import yaml

from src.exception.exception import RiskyException
from src.logging.logger import logging

def read_yaml_file(file_path):
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise RiskyException(e, sys)

def write_yaml_file(file_path, content, replace=True):
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise RiskyException(e, sys)

def save_object(file_path, obj):
    try:
        logging.info("Entered save object method of mainUtils class")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info("Exited save object method of mainUtils class")
    except Exception as e:
        raise RiskyException(e, sys)

def load_object(file):
    try:
        logging.info("Entered load object method")
        if not os.path.exists(file):
            raise Exception(f"File does not exist: {file}")
        with open(file, "rb") as file_obj:
            print(file_obj)
            return pickle.load(file_obj)
    except Exception as e:
        raise RiskyException(e, sys)


def load_numpy_array_data(file):
    try:
        with open(file, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise RiskyException(e, sys)
