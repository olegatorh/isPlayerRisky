import os

#pipeline constants
ARTIFACT_DIR="Artifacts"
PIPELINE_NAME="IsPlayerRisky"
TARGET_COLUMN="is_risky_player"
SCHEMA_FILE_PATH = os.path.join("dataScheme", "schema.yaml")
RAW_DATA="raw_data.csv"

#data ingestion constants
DATA_INGESTION_DIR="data_ingestion"
DATA_INGESTION_COLLECTION_DIR="data_ingestion_collected"
DATA_INGESTION_INGESTED_DIR="ingested_data"
DATA_INGESTION_FEATURE_STORE_DIR="feature_store"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION=0.2

#data validation constants
DATA_VALIDATION_DIR="data_validation"
DATA_VALIDATION_VALID_DIR="validated_data"
DATA_VALIDATION_INVALID_DIR="invalid_data"
DATA_VALIDATION_DRIFT_REPORT="drift_report.yaml"
DATA_VALIDATION_DRIFT_REPORT_DIR="drift_report"

#data transformation constants
DATA_TRANSFORMATION_DIR="data_transformation"
TRANSFORMED_DATA_DIR="transformed_data"
PREPROCESSOR_DIR="preprocessor"
PREPROCESSOR = "preprocessing.pkl"


#model training
MODEL_FILE_NAME="model.pkl"
MODEL_TRAINED_DIR="trained_model"
MODEL_EXPECTED_SCORE=0.6
MODEL_OVER_FITTING_UNDER_FITTING_THRESHOLD=0.05

TRAIN_FILE_NAME="train.csv"
TEST_FILE_NAME="test.csv"
