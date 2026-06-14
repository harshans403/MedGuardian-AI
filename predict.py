import joblib
import pandas as pd

# Load trained model
model = joblib.load("models/diagnosis_model.pkl")

# Load dataset
data = pd.read_csv("symptom_data.csv")

# Get user symptoms
symptoms = input("Enter symptoms separated by commas: ")

# Convert commas to spaces
symptoms_processed = symptoms.replace(",", " ").lower()

# Predict disease
prediction = model.predict([symptoms_processed])

disease = prediction[0]

print("\n==============================")
print("AI HEALTHCARE ASSISTANT")
print("==============================")
print(f"Possible Disease: {disease}")

# Find disease information
result = data[data["disease"] == disease]

if not result.empty:
    specialist = result.iloc[0]["specialist"]
    emergency = result.iloc[0]["emergency"]

    print(f"Recommended Specialist: {specialist}")
    print(f"Emergency Level: {emergency}")
else:
    print("Specialist information not found.")

# Emergency symptom list
dangerous_symptoms = [
    "chest pain",
    "shortness of breath",
    "stroke",
    "unconscious",
    "severe bleeding",
    "heart attack"
]

# Check emergency symptoms
user_symptoms = [s.strip().lower() for s in symptoms.split(",")]

for symptom in user_symptoms:
    if symptom in dangerous_symptoms:
        print("\n🚨 EMERGENCY ALERT 🚨")
        print("Seek immediate medical attention!")
        break