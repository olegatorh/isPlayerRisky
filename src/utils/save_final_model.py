import os

from src.constants import FINAL_MODEL_DIR, PREPROCESSOR, FINAL_MODEL_NAME
from src.utils.main_utils import save_object, load_object


def save_final_model(model_file_path, preprocessor_file_path):
    print("FINAL_MODEL_DIR", FINAL_MODEL_DIR)
    os.makedirs(FINAL_MODEL_DIR, exist_ok=True)

    # preprocessor = load_object(preprocessor_file_path)
    model = load_object(model_file_path)

    # final_preprocessor_path = os.path.join(FINAL_MODEL_DIR, PREPROCESSOR)
    final_model_path = os.path.join(FINAL_MODEL_DIR, FINAL_MODEL_NAME)

    # save_object(final_preprocessor_path, preprocessor)
    save_object(final_model_path, model)