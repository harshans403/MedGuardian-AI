from flask import Flask, render_template, request
import pandas as pd
import joblib
import pdfplumber
import os
from werkzeug.utils import secure_filename

# =====================================================
# FLASK APP
# =====================================================

app = Flask(__name__)

# =====================================================
# UPLOAD FOLDER
# =====================================================

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# =====================================================
# LOAD MODEL
# =====================================================

try:
    model = joblib.load("models/diagnosis_model.pkl")
except Exception as e:
    print("Disease model error:", e)
    model = None

# =====================================================
# LOAD CSV
# =====================================================

try:
    data = pd.read_csv("symptom_data.csv")
except Exception as e:
    print("CSV error:", e)
    data = pd.DataFrame()

# =====================================================
# HOSPITAL DATABASE
# =====================================================

hospital_data = {
    "Cardiology": {
        "hospital": "Apollo Heart Institute",
        "department": "Heart Care Department"
    },
    "Neurology": {
        "hospital": "NIMHANS",
        "department": "Neurology Department"
    },
    "Orthopedics": {
        "hospital": "MIOT International",
        "department": "Orthopedic Department"
    },
    "Dermatology": {
        "hospital": "Apollo Hospitals",
        "department": "Skin Care Department"
    },
    "Gastroenterology": {
        "hospital": "AIG Hospitals",
        "department": "Digestive Health Department"
    },
    "General Medicine": {
        "hospital": "Apollo Hospitals",
        "department": "General Medicine Department"
    }
}

# =====================================================
# HOME ROUTE
# =====================================================

@app.route("/", methods=["GET", "POST"])
def home():

    disease = ""
    specialist = ""
    emergency = ""
    advice = ""

    hospital_department = ""
    recommended_hospital = ""

    report_text = ""

    if request.method == "POST":

        # =================================================
        # SYMPTOM ANALYSIS
        # =================================================

        symptoms = request.form.get("symptoms", "").strip()

        if symptoms and model is not None:

            try:
                prediction = model.predict([symptoms.lower()])
                disease = prediction[0]

                result = data[data["disease"] == disease]

                if not result.empty:

                    specialist = result.iloc[0]["specialist"]
                    emergency = result.iloc[0]["emergency"]

                    department = str(
                        result.iloc[0].get(
                            "department",
                            "General Medicine"
                        )
                    )

                    if department in hospital_data:

                        recommended_hospital = hospital_data[department]["hospital"]

                        hospital_department = hospital_data[department]["department"]

                advice = "Consult a healthcare professional."

            except Exception as e:
                disease = f"Prediction Error: {e}"

        # =================================================
        # PDF REPORT ANALYSIS
        # =================================================

        report = request.files.get("report")

        if report and report.filename:

            filename = secure_filename(report.filename)

            pdf_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                filename
            )

            report.save(pdf_path)

            try:

                extracted_pages = []

                with pdfplumber.open(pdf_path) as pdf:

                    for page in pdf.pages:

                        text = page.extract_text()

                        if text:
                            extracted_pages.append(text)

                report_text = "\n".join(extracted_pages)

                if not report_text.strip():

                    report_text = (
                        "No readable text found in the PDF. "
                        "Please upload a text-based PDF report."
                    )

            except Exception as e:

                report_text = (
                    f"Error reading PDF: {str(e)}"
                )

    return render_template(
        "index.html",
        disease=disease,
        specialist=specialist,
        emergency=emergency,
        advice=advice,
        hospital_department=hospital_department,
        recommended_hospital=recommended_hospital,
        report_text=report_text
    )

# =====================================================
# RUN APP
# =====================================================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=True
    )