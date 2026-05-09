from pymongo import MongoClient
from dotenv import load_dotenv
import gridfs
import tempfile
import os
import joblib
from tensorflow.keras.models import load_model
from transformers import AutoTokenizer, AutoModelForSequenceClassification

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "medical_diagnosis_ai")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
fs = gridfs.GridFS(db)


def load_file_from_gridfs(filename, suffix):
    file_doc = db.models.find_one({"name": filename})
    if not file_doc:
        print(f"File not found in MongoDB: {filename}")
        return None

    gridfs_file = fs.get(file_doc["gridfs_id"])
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.write(gridfs_file.read())
    temp_file.close()
    return temp_file.name


def load_lstm_from_mongo():
    model_path = load_file_from_gridfs("lstm_model.keras", ".keras")
    tokenizer_path = load_file_from_gridfs("lstm_tokenizer.pkl", ".pkl")
    encoder_path = load_file_from_gridfs("lstm_label_encoder.pkl", ".pkl")

    if not all([model_path, tokenizer_path, encoder_path]):
        return None, None, None

    model = load_model(model_path)
    tokenizer = joblib.load(tokenizer_path)
    encoder = joblib.load(encoder_path)
    return model, tokenizer, encoder


def load_rnn_from_mongo():
    model_path = load_file_from_gridfs("rnn_model.keras", ".keras")
    tokenizer_path = load_file_from_gridfs("rnn_tokenizer.pkl", ".pkl")
    encoder_path = load_file_from_gridfs("rnn_label_encoder.pkl", ".pkl")

    if not all([model_path, tokenizer_path, encoder_path]):
        return None, None, None

    model = load_model(model_path)
    tokenizer = joblib.load(tokenizer_path)
    encoder = joblib.load(encoder_path)
    return model, tokenizer, encoder


def load_baseline_from_mongo():
    model_path = load_file_from_gridfs("baseline_model.pkl", ".pkl")
    vectorizer_path = load_file_from_gridfs("baseline_vectorizer.pkl", ".pkl")
    encoder_path = load_file_from_gridfs("baseline_label_encoder.pkl", ".pkl")

    if not all([model_path, vectorizer_path, encoder_path]):
        return None, None, None

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    encoder = joblib.load(encoder_path)
    return model, vectorizer, encoder


def load_biobert_from_mongo():
    encoder_path = load_file_from_gridfs("biobert_label_encoder.pkl", ".pkl")
    if not encoder_path:
        return None, None, None

    encoder = joblib.load(encoder_path)

    if os.path.exists("biobert_model"):
        tokenizer = AutoTokenizer.from_pretrained("biobert_model")
        model = "bert-base-uncased"
        return model, tokenizer, encoder

    print("BioBERT model folder not found locally")
    return None, None, None


def load_all_models():
    print("Loading LSTM from MongoDB...")
    lstm_model, lstm_tokenizer, lstm_encoder = load_lstm_from_mongo()

    print("Loading RNN from MongoDB...")
    rnn_model, rnn_tokenizer, rnn_encoder = load_rnn_from_mongo()

    print("Loading Baseline from MongoDB...")
    baseline_model, baseline_vectorizer, baseline_encoder = load_baseline_from_mongo()

    print("Loading BioBERT...")
    biobert_model, biobert_tokenizer, biobert_encoder = load_biobert_from_mongo()

    return (
        lstm_model, lstm_tokenizer, lstm_encoder,
        rnn_model, rnn_tokenizer, rnn_encoder,
        baseline_model, baseline_vectorizer, baseline_encoder,
        biobert_model, biobert_tokenizer, biobert_encoder
    )