import os
import sys

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainingArtifact
from src.entity.config_entity import ModelTrainingConfig
from src.exception.exception import RiskyException
from src.utils.main_utils import load_object, save_object, load_numpy_array_data
from src.utils.ml_utils.classification_score import get_classification_score
from src.utils.ml_utils.estimator import RiskyModel
from src.utils.ml_utils.evaluator import evaluate_models


class ModelTraining:
    def __init__(self, model_training_config: ModelTrainingConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.model_training_config = model_training_config
        except Exception as e:
            raise RiskyException(e, sys)



    def train_model(self, x_train, y_train, x_test, y_test):
        try:
            models = {
                "Random Forest": RandomForestClassifier(verbose=1, random_state=42),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(verbose=1, random_state=42),
                "Logistic Regression": LogisticRegression(verbose=1, max_iter=1000, random_state=42),
                "AdaBoost": AdaBoostClassifier(random_state=42)
            }

            params = {

                "Decision Tree": {
                    "criterion": ["gini", "entropy"],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5],
                },

                "Random Forest": {
                    "n_estimators": [50, 100],
                    "max_depth": [None, 10],
                    "max_features": ["sqrt"],
                },

                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.05],
                    "n_estimators": [50, 100],
                    "subsample": [0.8, 1.0],
                },

                "Logistic Regression": {
                    "C": [0.1, 1, 10],
                    "solver": ["lbfgs"],
                    "max_iter": [100],
                },

                "AdaBoost": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.1, 0.05],
                }
            }

            model_report, fitted_models = evaluate_models(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                models=models,
                params=params
            )
            best_model_name = max(model_report, key=model_report.get)
            best_model = fitted_models[best_model_name]

            y_train_prediction = best_model.predict(x_train)
            classification_train_metric = get_classification_score(y_train, y_train_prediction)
            y_test_prediction = best_model.predict(x_test)
            classification_test_metric = get_classification_score(y_test, y_test_prediction)
            preprocessor = load_object(self.data_transformation_artifact.transformed_preprocessor_file_path)
            os.makedirs(self.model_training_config.model_dir, exist_ok=True)

            risky_model = RiskyModel(preprocessor=preprocessor, model=best_model)
            save_object(self.model_training_config.trained_model_file_path, risky_model)
            model_trainer_artifact = ModelTrainingArtifact(
                trained_model_file_path=self.model_training_config.trained_model_file_path,
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric
            )
            return model_trainer_artifact

        except Exception as e:
            raise RiskyException(e, sys)

    def initiate_model_training(self):
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train = train_arr[:, :-1]
            y_train = train_arr[:, -1]

            x_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]

            model_trainer_artifact = self.train_model(x_train, y_train, x_test, y_test)
            return model_trainer_artifact

        except Exception as e:
            raise RiskyException(e, sys)
