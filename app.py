from flask import Flask, render_template, request
from ultralytics import YOLO
import pandas as pd
import joblib
import pdfplumber
import os

app = Flask(__name__)

# ==========================================
# FOLDERS
# ==========================================

UPLOAD_FOLDER = "uploads"
ACCIDENT_FOLDER = "accident_images"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ACCIDENT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["ACCIDENT_FOLDER"] = ACCIDENT_FOLDER

# ==========================================
# LOAD DISEASE MODEL
# ==========================================

model = joblib.load("models/diagnosis_model.pkl")
data = pd.read_csv("symptom_data.csv")

# ==========================================
# LOAD AI MODELS
# ==========================================

fracture_model = YOLO(
    "runs/detect/train/weights/best.pt"
)

head_model = YOLO(
    "runs/detect/train-6/weights/best.pt"
)

bleeding_model = YOLO(
    "runs/detect/train-7/weights/best.pt"
)

burn_model = YOLO(
    "runs/classify/train-5/weights/best.pt"
)

# ==========================================
# HOSPITAL DATABASE
# ==========================================

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

# ==========================================
# INJURY DATABASE
# ==========================================

injury_data = {

    "Fracture": {
        "hospital": "Orthopedic Trauma Center",
        "department": "Emergency Orthopedics",
        "first_aid": "Immobilize the injured limb and avoid movement."
    },

    "Bleeding": {
        "hospital": "Trauma Hospital",
        "department": "Emergency Care",
        "first_aid": "Apply firm pressure on the wound immediately."
    },

    "Head Injury": {
        "hospital": "Neuroscience Hospital",
        "department": "Emergency Neurology",
        "first_aid": "Keep the patient still and seek emergency help."
    },

    "Burn": {
        "hospital": "Burn Care Center",
        "department": "Burn Unit",
        "first_aid": "Cool the burn under running water for 20 minutes."
    }
}

# ==========================================
# HOME
# ==========================================

@app.route("/", methods=["GET", "POST"])
def home():

    disease = ""
    specialist = ""
    emergency = ""
    advice = ""

    hospital_department = ""
    recommended_hospital = ""

    report_text = ""

    injury = ""
    injury_hospital = ""
    injury_department = ""
    first_aid = ""

    burn_grade = ""
    burn_confidence = ""

    if request.method == "POST":

        # ==================================
        # DISEASE PREDICTION
        # ==================================

        symptoms = request.form.get("symptoms", "")

        if symptoms:

            symptoms_processed = symptoms.lower()

            prediction = model.predict(
                [symptoms_processed]
            )

            disease = prediction[0]

            result = data[
                data["disease"] == disease
            ]

            if not result.empty:

                specialist = result.iloc[0]["specialist"]
                emergency = result.iloc[0]["emergency"]

                if specialist in hospital_data:

                    recommended_hospital = \
                        hospital_data[specialist]["hospital"]

                    hospital_department = \
                        hospital_data[specialist]["department"]

            advice = "Consult a healthcare professional."

        # ==================================
        # PDF REPORT
        # ==================================

        report = request.files.get("report")

        if report and report.filename:

            pdf_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                report.filename
            )

            report.save(pdf_path)

            try:

                with pdfplumber.open(pdf_path) as pdf:

                    pages = []

                    for page in pdf.pages:

                        txt = page.extract_text()

                        if txt:
                            pages.append(txt)

                    report_text = "\n".join(pages)

            except Exception as e:

                report_text = str(e)

        # ==================================
        # ACCIDENT IMAGE ANALYSIS
        # ==================================

        accident_image = request.files.get(
            "accident_image"
        )

        if accident_image and accident_image.filename:

            image_path = os.path.join(
                app.config["ACCIDENT_FOLDER"],
                accident_image.filename
            )

            accident_image.save(image_path)

            # -------------------------
            # FRACTURE
            # -------------------------

            fracture_results = fracture_model(
                image_path
            )

            if len(fracture_results[0].boxes) > 0:

                injury = "Fracture"

            # -------------------------
            # HEAD INJURY
            # -------------------------

            head_results = head_model(
                image_path
            )

            if len(head_results[0].boxes) > 0:

                injury = "Head Injury"

            # -------------------------
            # BLEEDING
            # -------------------------

            bleeding_results = bleeding_model(
                image_path
            )

            if len(bleeding_results[0].boxes) > 0:

                injury = "Bleeding"

            # -------------------------
            # BURN
            # -------------------------

            burn_results = burn_model(
                image_path
            )

            if burn_results[0].probs is not None:

                class_id = burn_results[0].probs.top1

                burn_grade = \
                    burn_results[0].names[class_id]

                burn_confidence = round(
                    float(
                        burn_results[0].probs.top1conf
                    ) * 100,
                    2
                )

                injury = "Burn"

            # -------------------------
            # FIRST AID
            # -------------------------

            if injury in injury_data:

                injury_hospital = \
                    injury_data[injury]["hospital"]

                injury_department = \
                    injury_data[injury]["department"]

                first_aid = \
                    injury_data[injury]["first_aid"]

    return render_template(

        "index.html",

        disease=disease,
        specialist=specialist,
        emergency=emergency,
        advice=advice,

        hospital_department=hospital_department,
        recommended_hospital=recommended_hospital,

        report_text=report_text,

        injury=injury,
        injury_hospital=injury_hospital,
        injury_department=injury_department,
        first_aid=first_aid,

        burn_grade=burn_grade,
        burn_confidence=burn_confidence
    )

# ==========================================
# RUN
# ==========================================

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 5000)
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )