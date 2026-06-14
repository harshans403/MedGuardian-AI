import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

os.makedirs("models", exist_ok=True)

data = pd.read_csv("symptom_data.csv")

model = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB())
])

model.fit(data["symptom"], data["disease"])

joblib.dump(model, "models/diagnosis_model.pkl")

print("Model trained successfully!")