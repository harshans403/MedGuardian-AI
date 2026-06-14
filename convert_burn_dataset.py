import os
import pandas as pd
import shutil

BASE_DIR = r"Dataset\burn\burn.v1i.multiclass"

for split in ["train", "valid", "test"]:

    split_folder = os.path.join(BASE_DIR, split)

    csv_file = os.path.join(split_folder, "_classes.csv")

    print(f"\nProcessing {split}...")
    print("CSV:", csv_file)

    if not os.path.exists(csv_file):
        print("ERROR: CSV file not found!")
        continue

    df = pd.read_csv(csv_file)

    # Remove spaces from column names
    df.columns = df.columns.str.strip()

    print("Columns:", df.columns.tolist())

    grade1_folder = os.path.join(split_folder, "gradei")
    grade2_folder = os.path.join(split_folder, "gradeii")

    os.makedirs(grade1_folder, exist_ok=True)
    os.makedirs(grade2_folder, exist_ok=True)

    copied_grade1 = 0
    copied_grade2 = 0

    for _, row in df.iterrows():

        image_name = str(row["filename"]).strip()

        source_image = os.path.join(split_folder, image_name)

        if not os.path.exists(source_image):
            continue

        try:
            grade1 = int(row["gradei"])
            grade2 = int(row["gradeii"])
        except:
            continue

        if grade1 == 1:
            shutil.copy2(
                source_image,
                os.path.join(grade1_folder, image_name)
            )
            copied_grade1 += 1

        if grade2 == 1:
            shutil.copy2(
                source_image,
                os.path.join(grade2_folder, image_name)
            )
            copied_grade2 += 1

    print(f"Grade I Images  : {copied_grade1}")
    print(f"Grade II Images : {copied_grade2}")

print("\nDataset conversion completed successfully!")