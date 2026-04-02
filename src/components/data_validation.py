import os
import sys

import pandas as pd
from pandas import read_csv

from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig
from src.exception.exception import RiskyException
from src.logging.logger import logging
from src.utils.main_utils import read_yaml_file, write_yaml_file


class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self.schema_config = read_yaml_file(data_validation_config.training_pipeline_config.schema_config)
        except Exception as e:
            raise RiskyException(e, sys)


    def validate_number_columns(self, dataframe: pd.DataFrame):
        try:
            number_columns = len(self.schema_config["columns"])
            if len(dataframe.columns) != number_columns:
                return False
            return True
        except Exception as e:
            raise RiskyException(e, sys)

    def validate_column_types(self, dataframe: pd.DataFrame) -> bool:
        try:
            expected_columns = self.schema_config["columns"]
            logging.info("Checking required column types")

            actual_dtypes = dataframe.dtypes.to_dict()

            for column_name, expected_dtype in expected_columns.items():
                if column_name not in dataframe.columns:
                    logging.error(f"Column '{column_name}' is missing in dataframe")
                    return False

                actual_dtype = str(actual_dtypes[column_name])

                if actual_dtype != expected_dtype:
                    logging.error(
                        f"Column '{column_name}' type mismatch. "
                        f"Expected: {expected_dtype}, Found: {actual_dtype}"
                    )
                    return False

            logging.info("All column types are valid")
            return True

        except Exception as e:
            raise RiskyException(e, sys)

    def validate_missing_values(self, dataframe: pd.DataFrame) -> bool:
        try:
            logging.info("Checking missing values in dataframe")

            missing_percentage = (dataframe.isnull().sum() / len(dataframe)) * 100
            missing_percentage = missing_percentage[missing_percentage > 0]

            if missing_percentage.empty:
                logging.info("No missing values found in dataframe")
                return True

            for column, percentage in missing_percentage.items():
                logging.warning(
                    f"Column '{column}' has {round(percentage, 2)}% missing values"
                )

            max_missing_threshold = 20.0
            for column, percentage in missing_percentage.items():
                if percentage > max_missing_threshold:
                    logging.error(
                        f"Column '{column}' exceeds missing value threshold. "
                        f"Missing: {round(percentage, 2)}%, Threshold: {max_missing_threshold}%"
                    )
                    return False

            logging.info("Missing value validation completed successfully")
            return True

        except Exception as e:
            raise RiskyException(e, sys)

    def data_drift_validation(self, base_df: pd.DataFrame, current_df: pd.DataFrame) -> dict:
        try:
            from scipy.stats import ks_2samp, chi2_contingency

            drift_report = {}

            numerical_columns = self.schema_config["numerical_columns"]
            categorical_columns = self.schema_config["categorical_columns"]

            for column in numerical_columns:
                if column == self.schema_config["target_column"]:
                    continue

                base_data = base_df[column].dropna()
                current_data = current_df[column].dropna()

                if len(base_data) == 0 or len(current_data) == 0:
                    drift_report[column] = {
                        "drift_status": False,
                        "reason": "empty_column_after_dropna"
                    }
                    continue

                ks_stat, p_value = ks_2samp(base_data, current_data)

                drift_report[column] = {
                    "type": "numerical",
                    "p_value": float(p_value),
                    "drift_status": bool(p_value < 0.05)
                }

            for column in categorical_columns:
                base_dist = base_df[column].fillna("MISSING").value_counts()
                current_dist = current_df[column].fillna("MISSING").value_counts()

                all_categories = list(set(base_dist.index).union(set(current_dist.index)))

                base_freq = [base_dist.get(cat, 0) for cat in all_categories]
                current_freq = [current_dist.get(cat, 0) for cat in all_categories]

                contingency_table = [base_freq, current_freq]
                chi2, p_value, _, _ = chi2_contingency(contingency_table)

                drift_report[column] = {
                    "type": "categorical",
                    "p_value": float(p_value),
                    "drift_status": bool(p_value < 0.05)
                }
            drift_report_file_path = self.data_validation_config.drift_report_file
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)
            write_yaml_file(drift_report_file_path, drift_report)
            return drift_report

        except Exception as e:
            raise RiskyException(e, sys)

    def initiate_data_validation(self):
        try:
            train_df = pd.read_csv(self.data_ingestion_artifact.trained_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            columns_status = self.validate_number_columns(train_df)
            logging.info(f"Validation status of number of columns: {columns_status}")

            column_types_status = self.validate_column_types(train_df)
            logging.info(f"Validation status of column types: {column_types_status}")

            missing_values_status = self.validate_missing_values(train_df)
            logging.info(f"Validation status of missing values: {missing_values_status}")

            drift_status = self.data_drift_validation(train_df, test_df)
            logging.info(f"Data drift status: {drift_status}")

            validation_status = all([
                columns_status,
                column_types_status,
                missing_values_status
            ])

            os.makedirs(self.data_validation_config.data_validation_dir, exist_ok=True)

            valid_train_file = None
            valid_test_file = None
            invalid_train_file = None
            invalid_test_file = None

            if validation_status:
                os.makedirs(self.data_validation_config.validated_data_dir, exist_ok=True)
                train_df.to_csv(self.data_validation_config.valid_train_file, index=False, header=True)
                test_df.to_csv(self.data_validation_config.valid_test_file, index=False, header=True)

                valid_train_file = self.data_validation_config.valid_train_file
                valid_test_file = self.data_validation_config.valid_test_file

                logging.info("Data validation completed successfully. Valid train/test files saved.")
            else:
                os.makedirs(self.data_validation_config.invalid_data_dir, exist_ok=True)
                train_df.to_csv(self.data_validation_config.invalid_train_file, index=False, header=True)
                test_df.to_csv(self.data_validation_config.invalid_test_file, index=False, header=True)

                invalid_train_file = self.data_validation_config.invalid_train_file
                invalid_test_file = self.data_validation_config.invalid_test_file

                logging.info("Data validation failed. Invalid train/test files saved.")

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                valid_train_file=valid_train_file,
                valid_test_file=valid_test_file,
                invalid_train_file=invalid_train_file,
                invalid_test_file=invalid_test_file,
                drift_report_file=self.data_validation_config.drift_report_file
            )

            return data_validation_artifact

        except Exception as e:
            raise RiskyException(e, sys)

