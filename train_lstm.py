from db import conditions_collection
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


MAX_WORDS = 5000
MAX_LEN = 120


def train_lstm():
    data = list(conditions_collection.find({}))

    texts = []
    labels = []

    for item in data:
        text = item.get("clean_text", "")
        label = item.get("condition", "")

        if text and label:
            texts.append(text)
            labels.append(label)

    if len(texts) < 2:
        print("Not enough data to train LSTM")
        return

    print(f"Training LSTM on {len(texts)} samples")

    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)

    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, maxlen=MAX_LEN, padding="post")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    y = to_categorical(y_encoded)

    num_classes = len(label_encoder.classes_)

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

    model.fit(X, y, epochs=30, batch_size=4, verbose=1)

    model.save("lstm_model.keras")
    joblib.dump(tokenizer, "lstm_tokenizer.pkl")
    joblib.dump(label_encoder, "lstm_label_encoder.pkl")

    print("LSTM model trained and saved")


if __name__ == "__main__":
    train_lstm()