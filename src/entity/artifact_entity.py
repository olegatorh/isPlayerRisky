from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    trained_file_path: str
    test_file_path: str


@dataclass
class DataValidationArtifact:
    validation_status: bool
    valid_train_file: str
    valid_test_file: str
    invalid_train_file: str
    invalid_test_file: str
    drift_report_file: str


@dataclass
class ClassificationMetricArtifact:
    f1_score: float
    precision_score: float
    recall_score: float


@dataclass
class DataTransformationArtifact:
    transformed_train_file_path: str
    transformed_test_file_path: str
    transformed_preprocessor_file_path: str


@dataclass
class ModelTrainingArtifact:
    trained_model_file_path: str
    train_metric_artifact: ClassificationMetricArtifact
    test_metric_artifact: ClassificationMetricArtifact
