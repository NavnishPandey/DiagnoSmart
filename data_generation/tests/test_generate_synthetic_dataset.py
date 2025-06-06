import pytest
from data_generation.generate_data import generate_synthetic_dataset, specialties, severity_indicators
import pandas as pd

def test_generate_dataset_shape():
    # Generate the synthetic dataset
    df = generate_synthetic_dataset(100)
    # Ensure required columns are present
    expected_columns = ["complaint", "specialty", "severity_level", "chronic"]
    assert all(col in df.columns for col in expected_columns), \
        f"Missing columns in dataset. Found: {df.columns}"

def test_specialty_distribution_evenness():
    n_samples = 110
    df = generate_synthetic_dataset(n_samples)

    # Get list of specialties used
    num_specialties = len(specialties)

    # Expected samples per specialty (integer division)
    expected_per_specialty = n_samples // num_specialties

    # Check that all specialties are present
    counts = df['specialty'].value_counts()
    assert set(counts.index) == set(specialties.keys()), \
        f"Not all specialties are represented. Found: {list(counts.index)}"

    # Verify that each specialty appears the expected number of times
    for specialty in specialties:
        assert counts[specialty] == expected_per_specialty, \
            f"Specialty '{specialty}' has {counts[specialty]} rows, expected {expected_per_specialty}."

# Ensure all specialties in the dataset are known and valid
def test_specialty_column_values():
    df = generate_synthetic_dataset(110)
    allowed_specialties = set(specialties.keys())
    assert set(df["specialty"]).issubset(allowed_specialties), \
        f"Found unknown specialties: {set(df['specialty']) - allowed_specialties}"

# Verify that all severity levels are valid and belong to the predefined categories
def test_severity_level_validity():
    df = generate_synthetic_dataset(200)
    valid_levels = set(severity_indicators.keys())
    assert df["severity_level"].isin(valid_levels).all(), \
        "Dataset contains invalid severity levels."

# Check that the 'chronic' column only contains boolean values
def test_chronic_column_boolean():
    df = generate_synthetic_dataset(50)
    assert df["chronic"].map(type).eq(bool).all(), "Non-boolean values found in 'chronic' column."

# Ensure there are no missing (NaN) values in the entire dataset
def test_no_missing_values():
    df = generate_synthetic_dataset(150)
    assert not df.isnull().any().any(), "Dataset contains missing values."

# Confirm that each complaint is a non-empty string
def test_complaint_is_nonempty_string():
    df = generate_synthetic_dataset(100)
    assert df["complaint"].apply(lambda x: isinstance(x, str) and len(x.strip()) > 0).all(), \
        "Some complaints are not valid non-empty strings."
