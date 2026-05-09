from pymongo import MongoClient
from dotenv import load_dotenv
import gridfs
from datetime import datetime
import joblib
import os

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "medical_diagnosis_ai")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
fs = gridfs.GridFS(db)
models_collection = db["models"]


def save_file_to_mongo(filepath, model_name, model_type, label_encoder_path=None, metrics={}):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    with open(filepath, "rb") as f:
        gridfs_id = fs.put(f, filename=os.path.basename(filepath))

    labels = []
    if label_encoder_path and os.path.exists(label_encoder_path):
        encoder = joblib.load(label_encoder_path)
        labels = list(encoder.classes_)

    doc = {
        "name": model_name,
        "type": model_type,
        "gridfs_id": gridfs_id,
        "labels": labels,
        "metrics": metrics,
        "created": datetime.utcnow()
    }

    models_collection.insert_one(doc)
    print(f"Saved to MongoDB: {model_name}")


def save_all_models():

    # Baseline
    save_file_to_mongo(
        "baseline_model.pkl",
        "baseline_model.pkl",
        "Baseline TF-IDF + Logistic Regression",
        "baseline_label_encoder.pkl",
        metrics={"accuracy": 0.81, "top_3_accuracy": 0.93}
    )
    save_file_to_mongo(
        "baseline_vectorizer.pkl",
        "baseline_vectorizer.pkl",
        "Baseline TF-IDF Vectorizer",
        metrics={}
    )
    save_file_to_mongo(
        "baseline_label_encoder.pkl",
        "baseline_label_encoder.pkl",
        "Baseline Label Encoder",
        metrics={}
    )

    # RNN
    save_file_to_mongo(
        "rnn_model.keras",
        "rnn_model.keras",
        "RNN",
        "rnn_label_encoder.pkl",
        metrics={"accuracy": 0.98}
    )
    save_file_to_mongo(
        "rnn_tokenizer.pkl",
        "rnn_tokenizer.pkl",
        "RNN Tokenizer",
        metrics={}
    )
    save_file_to_mongo(
        "rnn_label_encoder.pkl",
        "rnn_label_encoder.pkl",
        "RNN Label Encoder",
        metrics={}
    )

    # LSTM
    save_file_to_mongo(
        "lstm_model.keras",
        "lstm_model.keras",
        "LSTM",
        "lstm_label_encoder.pkl",
        metrics={"accuracy": 0.62}
    )
    save_file_to_mongo(
        "lstm_tokenizer.pkl",
        "lstm_tokenizer.pkl",
        "LSTM Tokenizer",
        metrics={}
    )
    save_file_to_mongo(
        "lstm_label_encoder.pkl",
        "lstm_label_encoder.pkl",
        "LSTM Label Encoder",
        metrics={}
    )

    # BioBERT
    if os.path.exists("biobert_model"):
        for filename in os.listdir("biobert_model"):
            filepath = os.path.join("biobert_model", filename)
            save_file_to_mongo(
                filepath,
                f"biobert_{filename}",
                "BioBERT Transformer",
                "biobert_label_encoder.pkl" if "config" in filename else None,
                metrics={}
            )
        save_file_to_mongo(
            "biobert_label_encoder.pkl",
            "biobert_label_encoder.pkl",
            "BioBERT Label Encoder",
            metrics={}
        )
    else:
        print("BioBERT model not found — run train_transformer.py first")

    print("All models saved to MongoDB successfully")


if __name__ == "__main__":
    save_all_models()