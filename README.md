# 🏥 Medical Diagnosis AI

An AI-powered medical diagnosis system that scrapes real NHS condition data, trains multiple machine learning models, and serves predictions through a Flask REST API with a browser interface.

> **Data Science & AI Graduation Project** — NHS Inform Medical Diagnosis System

---

## 📌 Overview

A user types free-text symptoms (e.g. *"I have a fever, sore throat and headache"*) and the system returns the top 3 most probable medical conditions with probabilities, warnings, and NHS recommendations — powered by 4 different AI models.

**Full Pipeline:**

```
NHS Inform A–Z  →  MongoDB  →  Preprocessing  →  AI Models  →  Flask API  →  Browser UI
```

---

## 🧠 AI Models

| Model | Accuracy | Precision | Recall | F1 Score | Top-3 Accuracy |
|-------|----------|-----------|--------|----------|----------------|
| **RNN** ⭐ | **95.6%** | **96.0%** | **95.6%** | **95.1%** | **97.3%** |
| Baseline (TF-IDF + LR) | 81.4% | 66.3% | 81.4% | 73.1% | 93.0% |
| LSTM | 62.3% | 60.9% | 62.3% | 61.3% | 63.9% |
| BERT Transformer | 41.9% | 38.5% | 41.9% | 39.2% | — |

> ⭐ **Best model: RNN** — highest accuracy across all metrics on this dataset size (183 conditions, 915 samples after augmentation).

### How Each Model Works

**1. Baseline — TF-IDF + Logistic Regression**
Converts symptom text into word-frequency vectors (TF-IDF with bigrams), then classifies using Logistic Regression. Simple, fast, and surprisingly strong (81.4% accuracy).

**2. RNN — Simple Recurrent Neural Network**
Reads symptom text word-by-word, maintaining a hidden memory state. Architecture: `Embedding(5000,64) → SimpleRNN(32) → Dense(64) → Dense(183, softmax)`. Data augmentation expanded 183 → 915 samples, enabling 95.6% accuracy.

**3. LSTM — Long Short-Term Memory**
An improved RNN with forget/input/output gates for better long-range memory. Architecture: `Embedding → LSTM(64) → Dropout(0.3) → Dense(64) → Dense(183, softmax)`. Underperformed due to dataset size — LSTM needs more data than available.

**4. BERT Transformer**
Fine-tuned `bert-base-uncased` (110M parameters) using HuggingFace Trainer for 3 epochs. Reads text bidirectionally for deep contextual understanding. Performance limited by dataset size — would significantly outperform all models with 10,000+ samples.

---

## 🗂️ Project Structure

```
medical-diagnosis-ai/
│
├── app.py                    # Flask API — all endpoints
├── db.py                     # MongoDB connection
├── scraper.py                # NHS Inform A–Z scraper
├── preprocess.py             # Text cleaning & normalization
│
├── train_baseline.py         # TF-IDF + Logistic Regression
├── train_rnn.py              # SimpleRNN with data augmentation
├── train_lstm.py             # LSTM with Dropout
├── train_lstm_metadata.py    # LSTM + saves metrics to MongoDB
├── train_transformer.py      # BERT fine-tuning (HuggingFace)
│
├── save_model_to_mongo.py    # Saves all models to MongoDB GridFS
├── load_model_from_mongo.py  # Loads all models from MongoDB GridFS
│
├── requirements.txt
├── .gitignore
└── README.md
```

> **Note:** Trained model files (`.keras`, `.pkl`, `biobert_model/`) are stored in MongoDB GridFS and are excluded from this repo. Run the training scripts to regenerate them.

---

## 🗄️ Database Design

### Conditions Collection
| Field | Description |
|-------|-------------|
| `condition` | Medical condition name (unique index) |
| `url` | Original NHS page URL |
| `symptoms` / `causes` / `warnings` / `recommendations` | Raw scraped text |
| `clean_symptoms` / `clean_causes` / `clean_warnings` / `clean_recommendations` | Cleaned versions |
| `clean_text` | All fields merged — used for model training |

### Models Collection (GridFS)
| Field | Description |
|-------|-------------|
| `name` | File name (e.g. `rnn_model.keras`) |
| `type` | Model type (RNN, LSTM, BERT, etc.) |
| `gridfs_id` | Reference to binary file in GridFS |
| `labels` | List of 183 condition names |
| `metrics` | Accuracy, Precision, Recall, F1, Top-3 |
| `created` | Timestamp |

---

## 🚀 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Browser UI — test all models interactively |
| `GET` | `/scrape?limit=N` | Scrape NHS Inform A–Z pages |
| `GET` | `/preprocess` | Clean and normalize all conditions in DB |
| `POST` | `/train` | Train one or all models `{"model": "rnn"}` |
| `POST` | `/save-models` | Save trained models to MongoDB GridFS |
| `GET` | `/conditions` | List all 183 conditions |
| `GET` | `/conditions/<name>` | Get one condition's full details |
| `POST` | `/predict` | LSTM prediction |
| `POST` | `/predict-rnn` | RNN prediction ⭐ |
| `POST` | `/predict-baseline` | Baseline TF-IDF prediction |
| `POST` | `/predict-transformer` | BERT prediction |
| `GET` | `/model-comparison` | Metrics for all saved models |

### Example Request
```bash
POST /predict-rnn
Content-Type: application/json

{
  "symptoms": "I have a fever, cough, and difficulty breathing",
  "age": 30,
  "gender": "male"
}
```

### Example Response
```json
{
  "model": "RNN",
  "predictions": [
    {
      "condition": "Pneumonia",
      "probability": 0.7823,
      "details": {
        "warnings": "Call 999 if breathing becomes very difficult...",
        "recommendations": "Rest, drink plenty of fluids..."
      }
    }
  ]
}
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/medical-diagnosis-ai.git
cd medical-diagnosis-ai
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up MongoDB
Create a `.env` file in the project root:
```
MONGO_URI=mongodb://localhost:27017
DB_NAME=medical_diagnosis
```

### 4. Scrape NHS data
```bash
# Via API after starting the server, or directly:
python scraper.py
```

### 5. Preprocess the data
```bash
python preprocess.py
```

### 6. Train the models
```bash
python train_rnn.py        # Best model
python train_baseline.py
python train_lstm.py
python train_transformer.py
```

### 7. Save models to MongoDB
```bash
python save_model_to_mongo.py
```

### 8. Run the API
```bash
python app.py
```

Open `http://localhost:5000` in your browser.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|-----------|---------|
| Python 3.13 | Main language |
| MongoDB + GridFS | Conditions database & model storage |
| Flask | REST API framework |
| BeautifulSoup + Requests | NHS web scraping |
| NLTK | Stop word removal |
| Scikit-learn | TF-IDF, Logistic Regression, metrics |
| TensorFlow / Keras | RNN and LSTM models |
| HuggingFace Transformers | BERT fine-tuning |
| PyTorch | BERT model backend |
| Joblib | Saving/loading `.pkl` files |
| python-dotenv | Secure environment variable loading |

---

## 📊 Dataset

- **Source:** [NHS Inform A–Z Conditions](https://www.nhsinform.scot/illnesses-and-conditions/a-to-z/)
- **Conditions scraped:** 183
- **Samples after augmentation:** 915 (5 variations per condition)
- **Fields extracted:** Symptoms, Causes, Warnings, Recommendations

---

## 📄 License

This project was developed as a Data Science & AI graduation project. NHS Inform content belongs to NHS Scotland.
