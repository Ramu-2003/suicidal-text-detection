from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from backend.model_pipeline import load_models

app = Flask(__name__)
CORS(app)

# DO NOT TRAIN ON RENDER — Only load models
print("Loading models...")
models = load_models()
print(f"Available models: {list(models.keys())}")


@app.route("/models", methods=["GET"])
def get_models():
    result = {}
    for name, data in models.items():
        result[name] = {"accuracy": round(data['accuracy'], 2)}
    return jsonify(result)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text")
    model_name = data.get("model")

    if not text or not model_name:
        return jsonify({"error": "Text and model are required"}), 400

    if model_name not in models:
        return jsonify({"error": "Model not found"}), 400

    model_data = models[model_name]
    vectorizer = model_data["vectorizer"]
    model = model_data["model"]
    accuracy = model_data["accuracy"]

    text_vect = vectorizer.transform([text])
    pred = model.predict(text_vect)[0]
    confidence = model.predict_proba(text_vect).max()

    label = "Suicidal" if pred == 1 else "Not Suicidal"

    return jsonify({
        "prediction": label,
        "model_used": model_name,
        "accuracy": round(accuracy, 2),
        "confidence": round(confidence * 100, 2)
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
