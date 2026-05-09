import os
import re
import joblib
import numpy as np
import torch

from flask import Flask, request, jsonify, render_template_string
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from db import conditions_collection
from scraper import scrape_all
from load_model_from_mongo import load_all_models
from preprocess import preprocess_all
from train_baseline import train_baseline
from train_rnn import train_rnn
from train_lstm import train_lstm
from train_transformer import train_transformer
from save_model_to_mongo import save_all_models

app = Flask(__name__)
MAX_LEN = 120


# ================= LOAD MODELS =================

(
    lstm_model, lstm_tokenizer, lstm_encoder,
    rnn_model, rnn_tokenizer, rnn_encoder,
    baseline_model, baseline_vectorizer, baseline_encoder,
    biobert_model, biobert_tokenizer, biobert_encoder
) = load_all_models()


# ================= HELPERS =================

def clean_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_condition_details(condition_name):
    return conditions_collection.find_one(
        {"condition": {"$regex": re.escape(condition_name.split("(")[0].strip()), "$options": "i"}},
        {"_id": 0}
    )


def neural_predict(model, tokenizer, encoder, text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")
    probs = model.predict(padded)[0]
    top_indices = np.argsort(probs)[-3:][::-1]

    results = []
    for i in top_indices:
        condition_name = encoder.inverse_transform([i])[0]
        probability = float(probs[i])
        results.append({
            "condition": condition_name,
            "probability": round(probability, 4),
            "details": get_condition_details(condition_name)
        })
    return results


def biobert_predict(text):
    inputs = biobert_tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128
    )
    with torch.no_grad():
        outputs = biobert_model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1)[0].numpy()
    top_indices = np.argsort(probs)[-3:][::-1]

    results = []
    for i in top_indices:
        condition_name = biobert_encoder.inverse_transform([i])[0]
        probability = float(probs[i])
        results.append({
            "condition": condition_name,
            "probability": round(probability, 4),
            "details": get_condition_details(condition_name)
        })
    return results


# ================= UI =================

@app.route("/")
def home():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Medical Diagnosis AI</title>
        <style>
            body { font-family: Arial; background: #f4f6f8; padding: 40px; }
            .container { background: white; padding: 30px; border-radius: 12px;
                max-width: 700px; margin: auto; box-shadow: 0 0 12px rgba(0,0,0,0.1); }
            textarea, input, select, button { width: 100%; padding: 12px;
                margin-top: 8px; margin-bottom: 15px; border-radius: 8px;
                border: 1px solid #ccc; font-size: 15px; box-sizing: border-box; }
            button { background: #007bff; color: white; border: none; cursor: pointer; }
            button:hover { background: #0056b3; }
            pre { background: #111; color: #00ff66; padding: 15px;
                border-radius: 8px; white-space: pre-wrap;
                max-height: 450px; overflow: auto; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Medical Diagnosis AI</h2>
            <label>Symptoms:</label>
            <textarea id="symptoms" rows="4">I have fever and cough</textarea>
            <label>Age:</label>
            <input type="number" id="age" value="30">
            <label>Gender:</label>
            <select id="gender">
                <option value="male">Male</option>
                <option value="female">Female</option>
            </select>
            <label>Choose Model:</label>
            <select id="model">
                <option value="/predict">LSTM</option>
                <option value="/predict-rnn">RNN</option>
                <option value="/predict-baseline">Baseline TF-IDF + Logistic Regression</option>
                <option value="/predict-transformer">BioBERT Transformer</option>
            </select>
            <button onclick="predict()">Predict</button>
            <h3>Result:</h3>
            <pre id="result">Result will appear here...</pre>
        </div>
        <script>
            async function predict() {
                const endpoint = document.getElementById("model").value;
                const data = {
                    symptoms: document.getElementById("symptoms").value,
                    age: parseInt(document.getElementById("age").value),
                    gender: document.getElementById("gender").value
                };
                const response = await fetch(endpoint, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(data)
                });
                const result = await response.json();
                document.getElementById("result").textContent =
                    JSON.stringify(result, null, 2);
            }
        </script>
    </body>
    </html>
    """)


# ================= API INFO =================

@app.route("/api")
def api_info():
    return jsonify({
        "message": "Medical Diagnosis AI API is running",
        "routes": [
            "/scrape",
            "/preprocess",
            "/train",
            "/save-models",
            "/conditions",
            "/conditions/<condition_name>",
            "/predict",
            "/predict-rnn",
            "/predict-baseline",
            "/predict-transformer",
            "/model-comparison"
        ]
    })


# ================= SCRAPE =================

@app.route("/scrape", methods=["GET"])
def scrape():
    try:
        limit = request.args.get("limit", default=10, type=int)
        data = scrape_all(limit=limit)
        return jsonify({
            "message": "Scraping completed successfully",
            "inserted_or_updated": len(data)
        })
    except Exception as e:
        return jsonify({"error": str(e), "message": "Scraping failed"}), 500


# ================= PREPROCESS =================

@app.route("/preprocess", methods=["GET"])
def preprocess_data():
    try:
        preprocess_all()
        count = conditions_collection.count_documents({})
        return jsonify({
            "message": "Preprocessing completed",
            "processed": count
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ================= TRAIN =================

@app.route("/train", methods=["POST"])
def train():
    try:
        data = request.get_json()
        model_type = data.get("model", "all")

        results = {}

        if model_type in ("baseline", "all"):
            train_baseline()
            results["baseline"] = "trained"

        if model_type in ("rnn", "all"):
            train_rnn()
            results["rnn"] = "trained"

        if model_type in ("lstm", "all"):
            train_lstm()
            results["lstm"] = "trained"

        if model_type in ("transformer", "all"):
            train_transformer()
            results["transformer"] = "trained"

        return jsonify({
            "message": "Training completed",
            "trained": results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ================= SAVE MODELS =================

@app.route("/save-models", methods=["POST"])
def save_models():
    try:
        save_all_models()
        return jsonify({
            "message": "All models saved to MongoDB successfully"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ================= CONDITIONS =================

@app.route("/conditions", methods=["GET"])
def get_conditions():
    conditions = list(conditions_collection.find({}, {"_id": 0}))
    return jsonify({"count": len(conditions), "data": conditions})


@app.route("/conditions/<condition_name>", methods=["GET"])
def get_condition(condition_name):
    condition = get_condition_details(condition_name)
    if not condition:
        return jsonify({"error": "Condition not found"}), 404
    return jsonify(condition)


# ================= PREDICT LSTM =================

@app.route("/predict", methods=["POST"])
def predict_lstm():
    if not lstm_model or not lstm_tokenizer or not lstm_encoder:
        return jsonify({"error": "LSTM model not loaded"}), 404

    data = request.get_json()
    text = data.get("symptoms", "")
    if not text:
        return jsonify({"error": "No symptoms provided"}), 400

    results = neural_predict(lstm_model, lstm_tokenizer, lstm_encoder, text)
    return jsonify({"model": "LSTM", "input": data, "predictions": results})


# ================= PREDICT RNN =================

@app.route("/predict-rnn", methods=["POST"])
def predict_rnn():
    if not rnn_model or not rnn_tokenizer or not rnn_encoder:
        return jsonify({"error": "RNN model not loaded"}), 404

    data = request.get_json()
    text = data.get("symptoms", "")
    if not text:
        return jsonify({"error": "No symptoms provided"}), 400

    results = neural_predict(rnn_model, rnn_tokenizer, rnn_encoder, text)
    return jsonify({"model": "RNN", "input": data, "predictions": results})


# ================= PREDICT BASELINE =================

@app.route("/predict-baseline", methods=["POST"])
def predict_baseline():
    if not baseline_model or not baseline_vectorizer or not baseline_encoder:
        return jsonify({"error": "Baseline model not loaded"}), 404

    data = request.get_json()
    text = data.get("symptoms", "")
    if not text:
        return jsonify({"error": "No symptoms provided"}), 400

    clean_input = clean_text(text)
    X = baseline_vectorizer.transform([clean_input])
    probs = baseline_model.predict_proba(X)[0]
    top_indices = np.argsort(probs)[-3:][::-1]

    results = []
    for i in top_indices:
        condition_name = baseline_encoder.inverse_transform([i])[0]
        probability = float(probs[i])
        results.append({
            "condition": condition_name,
            "probability": round(probability, 4),
            "details": get_condition_details(condition_name)
        })

    return jsonify({
        "model": "Baseline TF-IDF + Logistic Regression",
        "input": data,
        "predictions": results
    })


# ================= PREDICT TRANSFORMER =================

@app.route("/predict-transformer", methods=["POST"])
def predict_transformer():
    if not biobert_model or not biobert_tokenizer or not biobert_encoder:
        return jsonify({"error": "BioBERT model not loaded"}), 404

    data = request.get_json()
    text = data.get("symptoms", "")
    if not text:
        return jsonify({"error": "No symptoms provided"}), 400

    results = biobert_predict(text)
    return jsonify({
        "model": "BioBERT Transformer",
        "input": data,
        "predictions": results
    })


# ================= MODEL COMPARISON =================

@app.route("/model-comparison", methods=["GET"])
def model_comparison():
    results = []
    for doc in db.models.find({}, {"_id": 0, "name": 1, "type": 1, "metrics": 1, "created": 1}):
        results.append({
            "name": doc.get("name"),
            "type": doc.get("type"),
            "metrics": doc.get("metrics", {}),
            "created": str(doc.get("created", ""))
        })
    return jsonify({"models": results})


# ================= RUN =================

if __name__ == "__main__":
    app.run(debug=False)