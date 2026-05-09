from db import conditions_collection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, precision_score, recall_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
import numpy as np
import joblib

MAX_LEN = 120
VOCAB_SIZE = 5000

def augment_text(text):
    return [
        text,
        text.lower(),
        text.replace(" and ", " "),
        text.replace(",", ""),
        text.replace(".", "")
    ]

def train_rnn():
    data = list(conditions_collection.find({}))

    texts = []
    labels = []

    print("Total records from DB:", len(data))

    for item in data:
        condition = item.get("condition", "")
        symptoms = item.get("symptoms", "")
        if condition and symptoms:
            for aug in augment_text(symptoms):
                texts.append(aug)
                labels.append(condition)

    print("Samples after augmentation:", len(texts))

    if len(texts) == 0:
        print("No data found!")
        return

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    tokenizer = Tokenizer(num_words=VOCAB_SIZE)
    tokenizer.fit_on_texts(texts)
    X = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(X, maxlen=MAX_LEN)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Train size:", len(X_train))
    print("Classes:", len(set(y)))

    model = Sequential([
        Embedding(VOCAB_SIZE, 64, input_length=MAX_LEN),
        SimpleRNN(32),
        Dense(64, activation="relu"),
        Dense(len(set(y)), activation="softmax")
    ])

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=8,
        validation_data=(X_test, y_test)
    )

    loss, acc = model.evaluate(X_test, y_test)

    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    precision = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    recall = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    top_k_correct = 0
    for i, probs in enumerate(y_pred_probs):
        top3 = np.argsort(probs)[-3:]
        if y_test[i] in top3:
            top_k_correct += 1
    top_k_accuracy = top_k_correct / len(y_test)

    print(f"Accuracy:        {acc:.4f}")
    print(f"Precision:       {precision:.4f}")
    print(f"Recall:          {recall:.4f}")
    print(f"F1 Score:        {f1:.4f}")
    print(f"Top-3 Accuracy:  {top_k_accuracy:.4f}")

    model.save("rnn_model.keras")
    joblib.dump(tokenizer, "rnn_tokenizer.pkl")
    joblib.dump(label_encoder, "rnn_label_encoder.pkl")

    print("RNN trained and saved successfully")

if __name__ == "__main__":
    train_rnn()