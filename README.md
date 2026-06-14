# 🏥 MedGuardian AI

## Intelligent Medical Diagnosis and Accident Injury Detection System

MedGuardian AI is an advanced healthcare assistance platform that combines Artificial Intelligence, Machine Learning, Computer Vision, and Medical Data Analysis to provide early disease prediction and accident injury detection.

The system helps users identify possible health conditions by analyzing symptoms, medical reports, and accident images. It also recommends suitable hospitals, specialist departments, emergency levels, and first-aid guidance.

---

## 🚀 Features

### Disease Prediction
- Predicts diseases from symptoms.
- Recommends specialists and hospitals.
- Provides emergency level assessment.

### Medical Report Analysis
- Upload PDF medical reports.
- Extracts and displays report text automatically.

### Fracture Detection
- Detects bone fractures using YOLOv8.
- Provides first-aid guidance.

### Bleeding Detection
- Detects bleeding injuries from accident images.
- Suggests emergency treatment recommendations.

### Head Injury Detection
- Detects scalp injuries and bruises.
- Recommends neurological care.

### Burn Detection & Classification
- Classifies burn severity.
- Displays confidence score.
- Provides burn first-aid instructions.

---

## 🛠 Technologies Used

- Python
- Flask
- Machine Learning
- YOLOv8
- Pandas
- Joblib
- PDFPlumber
- HTML/CSS
- Computer Vision

---

## 📂 Project Structure

```text
MedGuardian-AI/
│
├── app.py
├── requirements.txt
├── symptom_data.csv
│
├── models/
│   └── diagnosis_model.pkl
│
├── templates/
│   └── index.html
│
├── uploads/
├── accident_images/
│
├── runs/
│   ├── detect/
│   └── classify/
│
└── README.md