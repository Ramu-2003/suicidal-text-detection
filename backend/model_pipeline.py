import os
import pickle

# Base project directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(BASE_DIR, "backend", "models")


def load_models():
    """
    Load all PKL models from models folder
    """
    model_info = {}

    if not os.path.exists(MODEL_DIR):
        print("Models directory not found!")
        return model_info

    for file in os.listdir(MODEL_DIR):
        if file.endswith(".pkl"):
            with open(os.path.join(MODEL_DIR, file), "rb") as f:
                data = pickle.load(f)
                model_info[file.replace(".pkl", "")] = data

    return model_info
