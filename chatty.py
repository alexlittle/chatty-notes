import os
import json
import argparse
import base64
from copy import deepcopy
from pathlib import Path

import ollama


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate dementia-focused summaries from FHIR Bundles"
    )
    parser.add_argument(
        "-b",
        "--bundle",
        required=True,
        help="Input FHIR bundle JSON file",
    )
    parser.add_argument(
        "--llm-model",
        default="llama3.1:8b",
        help="Ollama model name",
    )
    return parser.parse_args()


def load_bundle(path):
    with open(path) as f:
        return json.load(f)


def extract_patient(bundle):
    for e in bundle.get("entry", []):
        if e.get("resource", {}).get("resourceType") == "Patient":
            return e["resource"]
    return {}


def extract_document_text(bundle):
    """
    Extract raw decoded text from all DocumentReferences,
    but DO NOT join everything blindly.
    """
    texts = []
    for e in bundle.get("entry", []):
        r = e.get("resource", {})
        if r.get("resourceType") != "DocumentReference":
            continue

        for c in r.get("content", []):
            data = c.get("attachment", {}).get("data")
            if not data:
                continue

            try:
                decoded = base64.b64decode(data).decode("utf-8", errors="ignore")
                texts.append(decoded)
            except Exception:
                pass

    return texts


def build_dementia_prompt(patient, notes):
    age = patient.get("birthDate", "unknown")
    gender = patient.get("gender", "unknown")

    # HARD FILTER: only keep cognitively relevant snippets
    keywords = [
        "memory",
        "confusion",
        "cognitive",
        "dementia",
        "forget",
        "disoriented",
        "poor judgment",
        "needs assistance",
    ]

    relevant = [
        n[:500]
        for n in notes
        if any(k in n.lower() for k in keywords)
    ]

    context = "\n".join(relevant[:10])  # HARD CAP

    return f"""
You are a clinician reviewing a longitudinal medical record.

Task:
Determine whether this patient shows evidence of dementia or
progressive cognitive impairment.

Focus ONLY on:
- Memory loss
- Confusion or disorientation
- Declining executive function
- Loss of independence
- Cognitive diagnoses or screenings

Ignore:
- Dental care
- Infections
- Medication instructions
- Billing or procedural text

Patient:
- Age: {age}
- Sex: {gender}

Relevant excerpts:
{context}

Write 1–2 short paragraphs (max 150 words).
If no evidence exists, say so clearly.
"""


def generate_summary(prompt, model):
    response = ollama.chat(
        model=model,
        options={
            "num_predict": 200  # critical: prevents long notes
        },
        messages=[
            {"role": "system", "content": "You are a medical reviewer."},
            {"role": "user", "content": prompt},
        ],
    )
    return response["message"]["content"]


def main():
    args = parse_arguments()
    bundle = load_bundle(args.bundle)
    output_bundle = deepcopy(bundle)

    patient = extract_patient(bundle)
    notes = extract_document_text(bundle)

    prompt = build_dementia_prompt(patient, notes)
    summary = generate_summary(prompt, args.llm_model)

    # Save ONE clean summary
    output_bundle["dementia_summary"] = summary

    os.makedirs("output", exist_ok=True)
    out_path = Path("output") / Path(args.bundle).name
    with open(out_path, "w") as f:
        json.dump(output_bundle, f, indent=2)

    print(f"Processed {args.bundle}")


if __name__ == "__main__":
    main()