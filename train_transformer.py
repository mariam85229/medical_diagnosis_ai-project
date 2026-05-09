from db import conditions_collection
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from datasets import Dataset
import numpy as np
import joblib

MODEL_NAME = "bert-base-uncased"


def augment_text(text):
    return [
        text,
        text.lower(),
        text.replace(" and ", " "),
        text.replace(",", ""),
        text.replace(".", "")
    ]


def train_transformer():
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
        label = item.get("condition", "")

        if text.strip() and label:
            for aug in augment_text(text):
                texts.append(aug)
                labels.append(label)

    print(f"Training BERT on {len(texts)} samples after augmentation")

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    num_labels = len(label_encoder.classes_)

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, encoded_labels, test_size=0.2, random_state=42
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

    train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels.tolist()})
    test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels.tolist()})

    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = float((predictions == labels).mean())
        precision = float(precision_score(labels, predictions, average="weighted", zero_division=0))
        recall = float(recall_score(labels, predictions, average="weighted", zero_division=0))
        f1 = float(f1_score(labels, predictions, average="weighted", zero_division=0))
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    training_args = TrainingArguments(
        output_dir="./biobert_output",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()

    eval_results = trainer.evaluate()

    print(f"Accuracy:   {eval_results.get('eval_accuracy', 'N/A')}")
    print(f"Precision:  {eval_results.get('eval_precision', 'N/A')}")
    print(f"Recall:     {eval_results.get('eval_recall', 'N/A')}")
    print(f"F1 Score:   {eval_results.get('eval_f1', 'N/A')}")
    print("BERT model saved to biobert_model/")

    model.save_pretrained("biobert_model")
    tokenizer.save_pretrained("biobert_model")
    joblib.dump(label_encoder, "biobert_label_encoder.pkl")


if __name__ == "__main__":
    train_transformer()