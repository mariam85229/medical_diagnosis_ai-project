from db import conditions_collection
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime
import gridfs
import os

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "medical_diagnosis_ai")

MAX_WORDS = 5000
MAX_LEN = 120


def augment_text(text):
    return [
        text,
        text.lower(),
        text.replace(" and ", " "),
        text.replace(",", ""),
        text.replace(".", "")
    ]


def train_lstm_with_metadata():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    fs = gridfs.GridFS(db)
    models_collection = db["models"]

    data = list(conditions_collection.find({}))

    texts = []
    labels = []

    for item in data:
        text = item.get("clean_text", "")
        label = item.get("condition", "")
        if text and label:
            for aug in augment_text(text):
                texts.append(aug)
                labels.append(label)

    if len(texts) < 2:
        print("Not enough data to train")
        return

    print(f"Training LSTM with metadata logging on {len(texts)} samples")

    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, maxlen=MAX_LEN, padding="post")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)
    y = to_categorical(y_encoded, num_classes=num_classes)

    X_train, X_test, y_train, y_test, y_train_enc, y_test_enc = train_test_split(
        X, y, y_encoded, test_size=0.2, random_state=42
    )

    model = Sequential([
        Embedding(input_dim=MAX_WORDS, output_dim=64, input_length=MAX_LEN),
        LSTM(64),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    model.fit(X_train, y_train, epochs=10, batch_size=8, verbose=1)

    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = y_test_enc

    accuracy = float((y_pred == y_true).mean())
    f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    precision = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    recall = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))

    top_k_correct = 0
    for i, probs in enumerate(y_pred_probs):
        top3 = np.argsort(probs)[-3:]
        if y_true[i] in top3:
            top_k_correct += 1
    top_k_accuracy = top_k_correct / len(y_true)

    loss = float(model.evaluate(X_test, y_test, verbose=0)[0])

    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Precision:       {precision:.4f}")
    print(f"Recall:          {recall:.4f}")
    print(f"F1 Score:        {f1:.4f}")
    print(f"Top-3 Accuracy:  {top_k_accuracy:.4f}")

    model.save("lstm_model.keras")
    joblib.dump(tokenizer, "lstm_tokenizer.pkl")
    joblib.dump(label_encoder, "lstm_label_encoder.pkl")

    metrics = {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "top_3_accuracy": round(top_k_accuracy, 4),
        "loss": round(loss, 4)
    }

    files_to_save = [
        ("lstm_model.keras", "LSTM"),
        ("lstm_tokenizer.pkl", "LSTM Tokenizer"),
        ("lstm_label_encoder.pkl", "LSTM Label Encoder"),
    ]

    for filename, model_type in files_to_save:
        with open(filename, "rb") as f:
            gridfs_id = fs.put(f, filename=filename)

        models_collection.insert_one({
            "name": filename,
            "type": model_type,
            "gridfs_id": gridfs_id,
            "labels": list(label_encoder.classes_),
            "metrics": metrics,
            "created": datetime.now(datetime.UTC) if hasattr(datetime, 'UTC') else datetime.utcnow()
        })
        print(f"Saved to MongoDB: {filename}")

    print("LSTM training with metadata logging completed successfully")


if __name__ == "__main__":
    train_lstm_with_metadata()