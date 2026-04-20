
import pandas as pd
import json
import subprocess

import glob

SYNTHEA_OUTPUT = "../synthea/output"
patients   = pd.read_csv(f"{SYNTHEA_OUTPUT}/csv/patients.csv")
conditions = pd.read_csv(f"{SYNTHEA_OUTPUT}/csv/conditions.csv")

# Who has dementia?
has_dementia = conditions[
    conditions["DESCRIPTION"].str.contains(
        "dementia|alzheimer", case=False, na=False
    )
]["PATIENT"].unique()

labels = patients[["Id"]].copy()
labels["dementia"] = labels["Id"].isin(has_dementia).astype(int)

print(labels["dementia"].value_counts())

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

print("Saved dementia_dataset.csv")


