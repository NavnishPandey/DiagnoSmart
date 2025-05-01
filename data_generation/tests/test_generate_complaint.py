import pytest
from data_generation.generate_data import generate_complaint, specialties,severity_indicators
import re

#Tests if the output of the function has the expected structure
def test_generate_complaint_for_all_specialties():
    for spec in specialties.keys():
        complaint, severity, chronic = generate_complaint(spec)
        assert isinstance(complaint, str)
        assert severity in severity_indicators
        assert chronic in [True, False]

#Verifies that if a generated complaint includes "and", 
# the phrase following it contains a valid secondary symptom from the specialty's known symptoms.
def test_generate_complaint_secondary_symptom_all_specialties():
    for spec in specialties:
        complaint, _, _ = generate_complaint(spec)

        if "and" in complaint.lower():
            # Combine all possible secondary symptoms (primary + overlapping + layman)
            symptom_keywords = specialties[spec]["primary_symptoms"] \
                               + specialties[spec].get("overlapping_symptoms", []) \
                               + specialties[spec].get("layman_terms", [])
    #Divides the text in two parts : ebfore and after the first and encountered
    #And takes only the second part
            after_and_part = complaint.lower().split("and", 1)[1]

            matched_symptoms = [
                symptom.lower() for symptom in symptom_keywords
                if re.search(r'\b' + re.escape(symptom.lower()) + r'\b', after_and_part)
            ]

            assert matched_symptoms, f"No known secondary symptom found after 'and' in complaint: {complaint}"


#This test ensures that the generated complaint for each specialty 
#includes at least one expected keyword, such as 
#a primary symptom, common complaint, overlapping symptom, or layman term.
def test_generate_complaint_contains_specialty_keywords_all():
    for spec in specialties:
        complaint, _, _ = generate_complaint(spec)
        keywords = (
            specialties[spec].get("primary_symptoms", []) +
            specialties[spec].get("common_complaints", []) +
            specialties[spec].get("overlapping_symptoms", []) +
            specialties[spec].get("layman_terms", [])
        )
        assert any(k.lower() in complaint.lower() for k in keywords), \
            f"No expected keywords found in complaint for specialty {spec}: {complaint}"

#Tests exception on UnknownSpecialty as input
def test_generate_complaint_invalid_specialty():
    with pytest.raises(KeyError):
        generate_complaint("kjohiyfutd")
