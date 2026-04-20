
import pandas as pd
import json
import subprocess

import glob

SYNTHEA_OUTPUT = "../synthea/output"
patients   = pd.read_csv(f"{SYNTHEA_OUTPUT}/csv/patients.csv")
conditions = pd.read_csv(f"{SYNTHEA_OUTPUT}/csv/conditions.csv")

# Who has diabetes?

has_diabetes = conditions[
    conditions["DESCRIPTION"].str.contains(
        "diabetes", case=False, na=False
    )
]["PATIENT"].unique()

labels = patients[["Id"]].copy()
labels["diabetes"] = labels["Id"].isin(has_diabetes).astype(int)

print(labels["diabetes"].value_counts())

# === Run chatty on a few patients ===
fhir_files = glob.glob(f"{SYNTHEA_OUTPUT}/fhir/*.json")

for f in fhir_files[:10]:
    subprocess.run(
        ["python", "chatty.py", "-b", f, "--llm-model", "llama3.1:8b"],
        check=True,
    )

# === Build dataset from summaries ===
records = []

for f in glob.glob("output/*.json"):
    with open(f) as fp:
        bundle = json.load(fp)

    patient_id = None
    for e in bundle.get("entry", []):
        if e.get("resource", {}).get("resourceType") == "Patient":
            patient_id = e["resource"]["id"]
            break

    summary = bundle.get("dementia_summary", "").strip()
    if patient_id and summary:
        records.append({
            "Id": patient_id,
            "note": summary
        })

notes_df = pd.DataFrame(records)

dataset = notes_df.merge(labels, on="Id", how="inner")
dataset = dataset[["Id", "note", "dementia"]]

dataset.to_csv("dementia_dataset.csv", index=False)

print("Saved diabetes_dataset.csv")


