from db import conditions_collection
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, top_k_accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np
from collections import Counter

def train_baseline():
    data = list(conditions_collection.find({}))

    texts = []
    labels = []

    for item in data:
        text = " ".join([
            item.get("clean_symptoms", ""),
            item.get("clean_causes", ""),
            item.get("clean_warnings", ""),
            item.get("clean_recommendations", "")
        ])
        label = item.get("category", "")
        if text and label:
            texts.append(text)
            labels.append(label)

    counts = Counter(labels)
    filtered_texts, filtered_labels = [], []
    for text, label in zip(texts, labels):
        if counts[label] >= 2:
            filtered_texts.append(text)
            filtered_labels.append(label)

    texts, labels = filtered_texts, filtered_labels

    print(f"Total samples: {len(texts)}")
    print(f"Total classes: {len(set(labels))}")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        texts, y, test_size=0.25, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    probs = model.predict_proba(X_test_vec)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    try:
        top3 = top_k_accuracy_score(y_test, probs, k=3, labels=model.classes_)
    except Exception as e:
        print("Top-3 error:", e)
        top3 = 0

    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Precision:       {precision:.4f}")
    print(f"Recall:          {recall:.4f}")
    print(f"F1 Score:        {f1:.4f}")
    print(f"Top-3 Accuracy:  {top3:.4f}")

    joblib.dump(model, "baseline_model.pkl")
    joblib.dump(vectorizer, "baseline_vectorizer.pkl")
    joblib.dump(label_encoder, "baseline_label_encoder.pkl")

    print("Baseline model saved successfully")

if __name__ == "__main__":
    train_baseline()