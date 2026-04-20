import pandas as pd
import json
import base64
import subprocess
import glob

SYNTHEA_OUTPUT = "../synthea/output"
patients   = pd.read_csv(f"{SYNTHEA_OUTPUT}/csv/patients.csv")
conditions = pd.read_csv(f"{SYNTHEA_OUTPUT}/csv/conditions.csv")

# Who has dementia?
has_dementia = conditions[
    conditions['DESCRIPTION'].str.contains(
        'dementia|alzheimer', case=False, na=False
    )
]['PATIENT'].unique()

# Final label dataframe
labels = patients[['Id']].copy()
labels['dementia'] = labels['Id'].isin(has_dementia).astype(int)

print(labels['dementia'].value_counts())


fhir_files = glob.glob(f"{SYNTHEA_OUTPUT}/fhir/*.json")

for idx, f in enumerate(fhir_files[0:3]):
    subprocess.run(["python", "chatty.py", "-b", f, "--llm-backend", "ollama", "--llm-model", "llama3.1:8b"])
    print(f"{idx} processed")


records = []
for fhir_file in glob.glob("output/*.json"):
    with open(fhir_file) as f:
        bundle = json.load(f)

    patient_id, note_text = None, []
    for entry in bundle.get('entry', []):
        r = entry.get('resource', {})

        if r.get('resourceType') == 'Patient':
            patient_id = r['id']

        if r.get('resourceType') == 'DocumentReference':
            for content in r.get('content', []):
                encoded = content.get('attachment', {}).get('data')

                if not encoded:
                    continue

                # Base64 that chatty.py produces back to plain text
                decoded_text = base64.b64decode(encoded).decode("utf-8")

                note_text.append(decoded_text)

    if patient_id:
        records.append({
            'Id': patient_id,
            'note': ' '.join(note_text)
        })


notes_df = pd.DataFrame(records)

dataset = notes_df.merge(labels, on='Id')
dataset = dataset[['Id', 'note', 'dementia']]  # just the three columns
dataset.to_csv("dementia_dataset.csv", index=False)
