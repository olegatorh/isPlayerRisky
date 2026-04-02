import sys

from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV

from src.exception.exception import RiskyException


def evaluate_models(x_train, y_train, x_test, y_test, models, params):
    try:
        report = {}
        fitted_models = {}

        for model_name, model in models.items():
            param = params[model_name]

            gs = GridSearchCV(model, param, cv=3, n_jobs=-1)
            gs.fit(x_train, y_train)

            best_model = gs.best_estimator_
            fitted_models[model_name] = best_model

            y_test_pred = best_model.predict(x_test)
            test_model_score = f1_score(y_test, y_test_pred)

            report[model_name] = test_model_score

        return report, fitted_models

    except Exception as e:
        raise RiskyException(e, sys)