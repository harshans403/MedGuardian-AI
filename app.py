from flask import Flask, render_template, request
import pandas as pd
import joblib
import pdfplumber
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# =====================================================
# FOLDERS
# =====================================================

UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# =====================================================
# LOAD DISEASE MODEL
# =====================================================

try:
    model = joblib.load("models/diagnosis_model.pkl")
except Exception as e:
    print("Disease model error:", e)
    model = None

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
        # DISEASE PREDICTION
        # =================================================

        symptoms = request.form.get("symptoms", "").strip()

        if symptoms and model is not None:

            try:

                prediction = model.predict(
                    [symptoms.lower()]
                )

                disease = prediction[0]

                result = data[
                    data["disease"] == disease
                ]

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

                        recommended_hospital = (
                            hospital_data[department]["hospital"]
                        )

                        hospital_department = (
                            hospital_data[department]["department"]
                        )

                advice = (
                    "Consult a healthcare professional."
                )

            except Exception as e:

                disease = f"Prediction Error: {e}"

        # =================================================
        # PDF REPORT ANALYSIS
        # =================================================

        report = request.files.get("report")

        if report and report.filename:

            filename = secure_filename(
                report.filename
            )

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

                report_text = "\n".join(
                    extracted_pages
                )

            except Exception as e:

                report_text = (
                    f"PDF Error: {str(e)}"
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

    port = int(
        os.environ.get("PORT", 5000)
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=True
    )