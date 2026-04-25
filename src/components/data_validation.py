import os
import sys

import pandas as pd
from pandas import read_csv
from pandas.api.types import (
    is_object_dtype,
    is_string_dtype,
    is_numeric_dtype,
    is_integer_dtype,
    is_float_dtype,
    is_bool_dtype,
)

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

    def validate_columns(self, dataframe: pd.DataFrame) -> tuple[bool, list[str]]:
        try:
            expected_columns = set(self.schema_config["columns"].keys())
            actual_columns = set(dataframe.columns)

            missing_columns = sorted(expected_columns - actual_columns)
            extra_columns = sorted(actual_columns - expected_columns)

            errors = []

            if missing_columns:
                errors.append(f"Missing columns: {missing_columns}")

            if extra_columns:
                errors.append(f"Unexpected columns: {extra_columns}")

            return len(errors) == 0, errors

        except Exception as e:
            raise RiskyException(e, sys)

    def _is_expected_dtype(self, series: pd.Series, expected_dtype: str) -> bool:
        expected_dtype = expected_dtype.lower()

        if expected_dtype in ("str", "string", "object", "categorical", "category"):
            return is_object_dtype(series) or is_string_dtype(series)

        if expected_dtype in ("int", "int64", "integer"):
            return is_integer_dtype(series)

        if expected_dtype in ("float", "float64", "double"):
            return is_float_dtype(series)

        if expected_dtype in ("number", "numeric"):
            return is_numeric_dtype(series)

        if expected_dtype in ("bool", "boolean"):
            return is_bool_dtype(series)

        return str(series.dtype).lower() == expected_dtype

    def validate_column_types(self, dataframe: pd.DataFrame) -> tuple[bool, list[str]]:
        try:
            expected_columns = self.schema_config["columns"]
            errors = []

            logging.info("Checking required column types")

            for column_name, expected_dtype in expected_columns.items():
                if column_name not in dataframe.columns:
                    errors.append(f"Column '{column_name}' is missing in dataframe")
                    continue

                if not self._is_expected_dtype(dataframe[column_name], expected_dtype):
                    errors.append(
                        f"Column '{column_name}' type mismatch. "
                        f"Expected: {expected_dtype}, Found: {dataframe[column_name].dtype}"
                    )

            if errors:
                for error in errors:
                    logging.error(error)
                return False, errors

            logging.info("All column types are valid")
            return True, []

        except Exception as e:
            raise RiskyException(e, sys)


    def validate_missing_values(self, dataframe: pd.DataFrame) -> tuple[bool, list[str], list[str]]:
        try:
            logging.info("Checking missing values in dataframe")

            warnings = []
            errors = []

            missing_percentage = (dataframe.isnull().sum() / len(dataframe)) * 100
            missing_percentage = missing_percentage[missing_percentage > 0]

            if missing_percentage.empty:
                logging.info("No missing values found in dataframe")
                return True, warnings, errors

            max_missing_threshold = self.schema_config.get("max_missing_threshold", 20.0)
            warning_missing_threshold = self.schema_config.get("warning_missing_threshold", 1.0)

            for column, percentage in missing_percentage.items():
                msg = f"Column '{column}' has {round(percentage, 2)}% missing values"

                if percentage > max_missing_threshold:
                    errors.append(
                        f"{msg}. Threshold exceeded: {max_missing_threshold}%"
                    )
                elif percentage > warning_missing_threshold:
                    warnings.append(msg)

            for warning in warnings:
                logging.warning(warning)

            for error in errors:
                logging.error(error)

            return len(errors) == 0, warnings, errors

        except Exception as e:
            raise RiskyException(e, sys)

    def validate_dataframe(self, dataframe: pd.DataFrame, df_name: str) -> dict:
        try:
            schema_status, schema_errors = self.validate_columns(dataframe)
            type_status, type_errors = self.validate_column_types(dataframe)
            missing_status, missing_warnings, missing_errors = self.validate_missing_values(dataframe)

            return {
                "dataframe_name": df_name,
                "status": all([schema_status, type_status, missing_status]),
                "warnings": missing_warnings,
                "errors": schema_errors + type_errors + missing_errors
            }

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

            train_validation = self.validate_dataframe(train_df, "train")
            test_validation = self.validate_dataframe(test_df, "test")

            logging.info(f"Train validation status: {train_validation['status']}")
            logging.info(f"Test validation status: {test_validation['status']}")

            drift_report = self.data_drift_validation(train_df, test_df)
            logging.info("Data drift validation completed")

            validation_status = train_validation["status"] and test_validation["status"]

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

                logging.error(f"Train validation errors: {train_validation['errors']}")
                logging.error(f"Test validation errors: {test_validation['errors']}")
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

