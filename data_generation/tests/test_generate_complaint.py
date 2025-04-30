import pytest
from data_generation.generate_data import generate_complaint
import pandas as pd

def test_generate_complaint_structure():
    complaint, severity_level, is_chronic = generate_complaint("Dermatology")
    assert complaint.startswith("I have")  # Verify that the complaint starts with 'I have'
    assert "rash" in complaint or "itching" in complaint  # Check that one of the symptoms is present
    assert any(severity in complaint for severity in ["mild", "moderate", "severe"])  # Verify that there is a severity level
    assert "for" in complaint  # Verify that the temporal aspect is included

def test_generate_complaint_no_missing_data():
    complaint, severity_level, is_chronic = generate_complaint("Cardiology")
    assert severity_level in ["low", "moderate", "high"]  # Test that severity is one of the three values
    assert isinstance(is_chronic, bool)  # Verify that is_chronic is a boolean

def test_generate_complaint_secondary_symptom():
    complaint, severity_level, is_chronic = generate_complaint("Neurology")
    assert "and" in complaint  # Verify that secondary symptom has been added
    assert "fatigue" in complaint or "dizziness" in complaint  # Verify that one secondary symptom is actually present