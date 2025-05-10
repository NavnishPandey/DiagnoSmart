import pytest
import random
import pandas as pd
from unittest.mock import patch
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Data_generation.generate_data import introduce_typos, generate_complaint, generate_synthetic_dataset

# Mock data for specialties and other global variables
specialties = {
    "Cardiology": {
        "primary_symptoms": ["chest pain", "shortness of breath"],
        "overlapping_symptoms": ["fatigue", "dizziness"],
        "common_complaints": ["chest pain while running", "breathlessness at night"],
        "layman_terms": ["heart issues", "breathing trouble"],
        "procedures": ["ECG", "stress test"]
    },
    "Neurology": {
        "primary_symptoms": ["headache", "numbness"],
        "overlapping_symptoms": ["fatigue", "sleep problems"],
        "common_complaints": ["migraine episodes", "tingling sensation"],
        "layman_terms": ["brain issues", "nerve problems"],
        "procedures": ["MRI", "CT scan"]
    }
}

severity_indicators = {
    "low": ["mild"],
    "moderate": ["moderate"],
    "high": ["severe"]
}

temporal_patterns = {
    "continuous": ["persistent", "constant"],
    "episodic": ["on and off", "intermittent"]
}

chronic_indicators = {
    True: ["for years", "for a long time"],
    False: ["since last week", "for a few days"]
}

time_periods = ["last week", "last month", "yesterday"]

modifiers = ["occasional", "frequent", "persistent"]

# Patch the global variables inside the test file
@pytest.fixture(autouse=True)
def patch_globals(monkeypatch):
    monkeypatch.setitem(globals(), 'specialties', specialties)
    monkeypatch.setitem(globals(), 'severity_indicators', severity_indicators)
    monkeypatch.setitem(globals(), 'temporal_patterns', temporal_patterns)
    monkeypatch.setitem(globals(), 'chronic_indicators', chronic_indicators)
    monkeypatch.setitem(globals(), 'time_periods', time_periods)
    monkeypatch.setitem(globals(), 'modifiers', modifiers)

# Test for introduce_typos
def test_introduce_typos_no_typo():
    text = "This is a simple sentence"
    output = introduce_typos(text, prob=0)  # prob=0 means no typos
    assert output == text

def test_introduce_typos_with_typo():
    text = "Testing typos here"
    output = introduce_typos(text, prob=1)  # prob=1 forces typo introduction
    assert output != text
    assert len(output.split()) == len(text.split())

# Test for generate_complaint
def test_generate_complaint_structure():
    specialty = "Cardiology"
    complaint, severity_level, is_chronic = generate_complaint(specialty)
    assert isinstance(complaint, str)
    assert severity_level in ["low", "moderate", "high"]
    assert isinstance(is_chronic, bool)
    assert specialty in specialties

def test_generate_complaint_content():
    specialty = "Neurology"
    complaint, severity_level, is_chronic = generate_complaint(specialty)
    assert "I have" in complaint or "I'm a" in complaint


def test_generate_synthetic_dataset_specialties(monkeypatch):
    dummy_specialties = {
        "Cardiology": {
            "primary_symptoms": ["chest pain"],
            "overlapping_symptoms": ["fatigue"],
            "common_complaints": ["chest pain while running"],
            "layman_terms": ["heart issues"],
            "procedures": ["ECG"]
        }
    }
    
    # Patch specialties directly
    monkeypatch.setitem(sys.modules['Data_generation.generate_data'].__dict__, 'specialties', dummy_specialties)

    df = generate_synthetic_dataset(8)
    print(df.columns)  # Debugging line
    assert not df.empty, "Generated dataframe is empty!"
    
    unique_specialties = set(df["specialty"])
    expected_specialties = set(dummy_specialties.keys())
    assert unique_specialties.issubset(expected_specialties)

