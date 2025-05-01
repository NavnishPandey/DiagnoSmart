import pytest
from data_generation.generate_data import generate_complaint, specialties,severity_indicators
import pandas as pd

def test_generate_complaint_for_all_specialties():
    for spec in specialties.keys():
        complaint, severity, chronic = generate_complaint(spec)
        assert isinstance(complaint, str)
        assert severity in severity_indicators
        assert chronic in [True, False]


def test_generate_complaint_no_missing_data():
    complaint, severity_level, is_chronic = generate_complaint("Cardiology")
    assert severity_level in ["low", "moderate", "high"]  # Test that severity is one of the three values
    assert isinstance(is_chronic, bool)  # Verify that is_chronic is a boolean

import re

def test_generate_complaint_secondary_symptom_all_specialties():
    for spec in specialties.keys():
        complaint, _, _ = generate_complaint(spec)

        # Ensure 'and' is present (indicates a secondary symptom)
        assert "and" in complaint.lower(), f"No 'and' found in complaint: {complaint}"

        # Combine all possible secondary symptoms (primary + overlapping + layman)
        symptom_keywords = specialties[spec]["primary_symptoms"] \
                           + specialties[spec].get("overlapping_symptoms", []) \
                           + specialties[spec].get("layman_terms", [])

        # Extract the portion of the complaint after the first 'and'
        after_and_part = complaint.lower().split("and", 1)[1]

        # Normalize and check for any symptom present after 'and'
        matched_symptoms = [
            symptom.lower() for symptom in symptom_keywords
            if re.search(r'\b' + re.escape(symptom.lower()) + r'\b', after_and_part)
        ]

        assert matched_symptoms, f"No known secondary symptom found after 'and' in complaint: {complaint}"

        # Optionally, you could also assert which symptoms were matched:
        print(f"Matched symptoms in '{complaint}': {matched_symptoms}")


def test_generate_complaint_contains_specialty_keywords():
    complaint, _, _ = generate_complaint("Dermatology")
    keywords = specialties["Dermatology"]["primary_symptoms"] + specialties["Dermatology"]["common_complaints"]
    assert any(k.lower() in complaint.lower() for k in keywords)
