import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

MODEL_DIR = "models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

CSV_PATH = "../dataset/boamente_dataset.csv"


def load_dataset():
    """
    Load CSV and prepare dataframe.
    Drops rows with missing text or target.
    """
    df = pd.read_csv(CSV_PATH)
    df = df[['text', 'target']]
    df = df.dropna(subset=['text', 'target'])  # remove NaN
    df['label'] = df['target'].astype(int)
    return df


def train_models():
    """
    Train TF-IDF + Logistic Regression models as placeholders.
    Save models as PKL files if accuracy >=85%
    """
    df = load_dataset()
    X = df['text']
    y = df['label']

    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=5000)
    X_vect = vectorizer.fit_transform(X)

    models = {
        "DistilBERT": LogisticRegression(max_iter=500),
        "XLM-R": LogisticRegression(max_iter=500),
        "BERTimbau": LogisticRegression(max_iter=500)
    }

    model_info = {}

    X_train, X_test, y_train, y_test = train_test_split(
        X_vect, y, test_size=0.2, random_state=42)

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred) * 100
        print(f"{name} accuracy: {acc:.2f}%")

        # Save model if accuracy >=85%
        if acc >= 85:
            file_path = os.path.join(MODEL_DIR, f"{name}.pkl")
            with open(file_path, "wb") as f:
                pickle.dump({
                    "model": model,
                    "vectorizer": vectorizer,
                    "accuracy": acc
                }, f)
            model_info[name] = {"accuracy": acc}
        else:
            print(f"{name} accuracy <85%, skipping save.")

    return model_info


def load_models():
    """
    Load all PKL models from models folder
    """
    model_info = {}
    for file in os.listdir(MODEL_DIR):
        if file.endswith(".pkl"):
            with open(os.path.join(MODEL_DIR, file), "rb") as f:
                data = pickle.load(f)
                model_info[file.replace(".pkl", "")] = data
    return model_info
