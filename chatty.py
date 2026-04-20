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


def build_diabetes_prompt(patient, notes):
    age = patient.get("birthDate", "unknown")
    gender = patient.get("gender", "unknown")

    # HARD FILTER: only keep cognitively relevant snippets

    keywords = [
        # core diabetes terms
        "diabetes",
        "hyperglycemia",
        "hypoglycemia",
        "blood sugar",
        "glucose",
        "a1c",
        "hba1c",

        # symptoms
        "polyuria",
        "polydipsia",
        "increased thirst",
        "frequent urination",
        "fatigue",

        # medications
        "metformin",
        "insulin",
        "glipizide",
        "glyburide",
        "sitagliptin",
        "liraglutide",
        "semaglutide",

        # care patterns
        "dietary counseling",
        "glucose monitoring",
        "endocrinology",
        "diabetic",
    ]

    relevant = [
        n[:400]
        for n in notes
        if any(k in n.lower() for k in keywords)
    ]

    # Step 2: fallback if no disease signal
    if relevant:
        selected = relevant[:10]
    else:
        # fallback: neutral longitudinal snippets
        selected = [
            n[:250]
            for n in notes[:10]
        ]

    context = "\n".join(selected)

    return f"""
You are summarizing a patient’s longitudinal medical record.


Below are excerpts from the record:
--------------------
{context}
--------------------


Write a short, factual summary focused on metabolic health and diabetes-related
care over time.

Rules:
- Do NOT state a diagnosis explicitly.
- Do NOT use medical conclusions or labels.
- Describe only observations, symptoms, treatments, and care patterns.

Focus on:
- Blood sugar management issues
- Symptoms such as increased thirst, urination, or fatigue
- Medication use related to glucose control
- Lifestyle or dietary counseling
- Emergency visits or hospitalizations related to metabolic issues
- Long-term management patterns

Ignore:
- Dental care
- Unrelated acute infections
- Administrative text

Write 4–6 factual paragraphs using neutral clinical language.

Do not speculate. Do not summarize unrelated medical history.

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

    prompt = build_diabetes_prompt(patient, notes)
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