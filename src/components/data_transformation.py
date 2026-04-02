import os
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact
from src.entity.config_entity import DataTransformationConfig
from src.exception.exception import RiskyException
from src.logging.logger import logging
from src.utils.main_utils import read_yaml_file, save_object
from src.constants import SCHEMA_FILE_PATH


class DataTransformation:
    def __init__(
        self,
        data_validation_artifact: DataValidationArtifact,
        data_transformation_config: DataTransformationConfig
    ):
        try:
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise RiskyException(e, sys)

    def get_data_transformer_object(self) -> ColumnTransformer:
        try:
            numerical_columns = self.schema_config["numerical_columns"]
            categorical_columns = self.schema_config["categorical_columns"]

            logging.info("Creating numerical pipeline")
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            logging.info("Creating categorical pipeline")
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info("Creating preprocessing object")
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise RiskyException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Starting data transformation")

            train_df = pd.read_csv(self.data_validation_artifact.valid_train_file)
            test_df = pd.read_csv(self.data_validation_artifact.valid_test_file)

            target_column = self.schema_config["target_column"]

            logging.info("Splitting input and target features")
            input_feature_train_df = train_df.drop(columns=[target_column])
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column])
            target_feature_test_df = test_df[target_column]

            preprocessing_obj = self.get_data_transformer_object()

            logging.info("Fitting preprocessing object on training data and transforming train/test")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            os.makedirs(self.data_transformation_config.data_transformation_dir, exist_ok=True)
            os.makedirs(self.data_transformation_config.transformed_data_dir, exist_ok=True)
            os.makedirs(self.data_transformation_config.preprocessor_dir, exist_ok=True)

            logging.info("Saving transformed train array")
            np.save(self.data_transformation_config.transformed_train_file, train_arr)

            logging.info("Saving transformed test array")
            np.save(self.data_transformation_config.transformed_test_file, test_arr)

            logging.info("Saving preprocessing object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_file,
                obj=preprocessing_obj
            )

            data_transformation_artifact = DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file,
                transformed_preprocessor_file_path=self.data_transformation_config.preprocessor_file
            )

            logging.info("Data transformation completed successfully")
            return data_transformation_artifact

        except Exception as e:
            raise RiskyException(e, sys)