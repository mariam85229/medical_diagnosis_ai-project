import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "medical_diagnosis_ai")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

conditions_collection = db["conditions"]

conditions_collection.create_index("condition", unique=True)