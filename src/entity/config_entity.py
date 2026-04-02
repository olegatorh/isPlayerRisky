import datetime
import os

from src import constants


class TrainingPipelineConfig:
    def __init__(self, timestamp=datetime.datetime.now()):
        timestamp = timestamp.strftime("%Y:%m:%d|%H:%M:%S")
        self.timestamp = timestamp
        self.schema_config = constants.SCHEMA_FILE_PATH
        self.pipeline_name = constants.PIPELINE_NAME
        self.artifact_dir = constants.ARTIFACT_DIR
        self.artifact_path = os.path.join(self.artifact_dir, self.timestamp)

        self.train_test_split_ratio = constants.DATA_INGESTION_TRAIN_TEST_SPLIT_RATION
        self.train_file_name = constants.TRAIN_FILE_NAME
        self.test_file_name = constants.TEST_FILE_NAME



class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.training_pipeline_config = training_pipeline_config

        self.data_ingestion_dir = os.path.join(
            self.training_pipeline_config.artifact_path,
            constants.DATA_INGESTION_DIR
        )

        self.feature_store_dir = os.path.join(
            self.data_ingestion_dir,
            constants.DATA_INGESTION_FEATURE_STORE_DIR,
            constants.RAW_DATA
        )

        self.training_file_path = os.path.join(
            self.data_ingestion_dir,
            constants.DATA_INGESTION_INGESTED_DIR,
            self.training_pipeline_config.train_file_name
        )
        self.testing_file_path = os.path.join(
            self.data_ingestion_dir,
            constants.DATA_INGESTION_INGESTED_DIR,
            self.training_pipeline_config.test_file_name
        )

class DataValidationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.training_pipeline_config = training_pipeline_config

        self.data_validation_dir = os.path.join(
            self.training_pipeline_config.artifact_path,
            constants.DATA_VALIDATION_DIR
        )
        self.validated_data_dir = os.path.join(
            self.data_validation_dir,
            constants.DATA_VALIDATION_VALID_DIR
        )
        self.valid_train_file = os.path.join(
            self.validated_data_dir,
            self.training_pipeline_config.train_file_name,
        )
        self.valid_test_file = os.path.join(
            self.validated_data_dir,
            self.training_pipeline_config.test_file_name,
        )

        self.invalid_data_dir = os.path.join(
            self.data_validation_dir,
            constants.DATA_VALIDATION_INVALID_DIR,
        )
        self.invalid_train_file = os.path.join(
            self.invalid_data_dir,
            self.training_pipeline_config.train_file_name,
        )
        self.invalid_test_file = os.path.join(
            self.invalid_data_dir,
            self.training_pipeline_config.test_file_name,
        )

        self.drift_report_file = os.path.join(
            self.data_validation_dir,
            constants.DATA_VALIDATION_DRIFT_REPORT_DIR,
            constants.DATA_VALIDATION_DRIFT_REPORT
        )

class DataTransformationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.training_pipeline_config = training_pipeline_config

        self.data_transformation_dir = os.path.join(
            self.training_pipeline_config.artifact_path,
            constants.DATA_TRANSFORMATION_DIR
        )
        self.transformed_data_dir = os.path.join(
            self.data_transformation_dir,
            constants.TRANSFORMED_DATA_DIR
        )
        self.transformed_train_file = os.path.join(
            self.transformed_data_dir,
            constants.TRAIN_FILE_NAME.replace("csv", "npy")
        )
        self.transformed_test_file = os.path.join(
            self.transformed_data_dir,
            constants.TEST_FILE_NAME.replace("csv", "npy")
        )

        self.preprocessor_dir = os.path.join(
            self.data_transformation_dir,
            constants.PREPROCESSOR_DIR
        )
        self.preprocessor_file = os.path.join(
            self.preprocessor_dir,
            constants.PREPROCESSOR
        )

class ModelTrainingConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.training_pipeline_config = training_pipeline_config

        self.model_dir = os.path.join(
            self.training_pipeline_config.artifact_path,
            constants.MODEL_TRAINED_DIR
        )
        self.trained_model_file_path = os.path.join(
            self.model_dir,
            constants.MODEL_FILE_NAME
        )
        self.expected_accuracy: float = constants.MODEL_EXPECTED_SCORE
        self.overfitting_underfitting_threshold = constants.MODEL_OVER_FITTING_UNDER_FITTING_THRESHOLD


