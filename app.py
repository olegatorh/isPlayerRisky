from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.data_validation import DataValidation
from src.components.model_training import ModelTraining
from src.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig, DataValidationConfig, \
    DataTransformationConfig, ModelTrainingConfig
from src.logging.logger import logging
from src.utils.save_final_model import save_final_model

if __name__ == "__main__":
    logging.info("Starting Configuration")
    training_pipeline_config = TrainingPipelineConfig()
    logging.info("Starting Data Ingestion")
    data_ingestion_config = DataIngestionConfig(training_pipeline_config)
    data_ingestion = DataIngestion(data_ingestion_config)
    data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
    logging.info("Data ingestion finished")
    logging.info("Start Data Validation")
    data_validation_config = DataValidationConfig(training_pipeline_config)
    data_validation = DataValidation(data_ingestion_artifact, data_validation_config)
    data_validation_artefact = data_validation.initiate_data_validation()
    logging.info("Data Validation finished")
    logging.info("Data Transformation start")
    data_transformation_config = DataTransformationConfig(training_pipeline_config)
    data_transformation = DataTransformation(data_validation_artefact, data_transformation_config)
    data_transformation_artifact = data_transformation.initiate_data_transformation()
    logging.info("Data Transformation finished")
    logging.info("Start Model Training")
    model_training_config = ModelTrainingConfig(training_pipeline_config)
    model_training = ModelTraining(model_training_config, data_transformation_artifact)
    model_training_artifact = model_training.initiate_model_training()
    logging.info("Finish Model Training")
    logging.info("Finish Model Saving start")
    save_final_model(model_training_artifact.trained_model_file_path,
                     data_transformation_artifact.transformed_preprocessor_file_path)
    logging.info("Finish Model Saving finished")