import re
import nltk
from db import conditions_collection

# تحميل stopwords مرة واحدة
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))


def clean_text(text):
    if not text:
        return ""

    # lowercase
    text = text.lower()

    # remove punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # tokenize
    words = text.split()

    # remove stopwords
    words = [w for w in words if w not in stop_words]

    return " ".join(words)


def preprocess_all():
    data = list(conditions_collection.find({}))

    if not data:
        print("❌ No data found in MongoDB")
        return

    for item in data:
        combined_text = " ".join([
            item.get("symptoms", ""),
            item.get("causes", ""),
            item.get("warnings", ""),
            item.get("recommendations", "")
        ])

        cleaned = clean_text(combined_text)

        conditions_collection.update_one(
            {"_id": item["_id"]},
            {"$set": {"clean_text": cleaned}}
        )

        print("Processed:", item.get("condition"))

    print("✅ Preprocessing completed")

if __name__ == "__main__":
    preprocess_all()