import pytest
import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Data_generation.generate_data import generate_complaint, specialties,severity_indicators
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
    #Search any sympom_keyword in after_and_part using regulare expressions to
    #handle possible issues like special chars and then stores the ones found
    #in the list matched_symptoms
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

#Test the generation of optional parts in a base complaint
def test_generate_complaint_optional_components():
    for spec in specialties:
        found = {
            "severity": False,
            "temporal": False,
            "secondary_symptom": False,
            "procedure": False,
            "age_gender": False,
            "medical_history": False
        }

        # Execute several attempts, 5 for instance
        for _ in range(5):
            complaint, severity_level, is_chronic = generate_complaint(spec)
            lower = complaint.lower()

            # Check for severity
            if any(sev in lower for sev in sum(severity_indicators.values(), [])):
                found["severity"] = True

            # Check for temporal expression
            if re.search(r"\b(for|since|over|past)\b", lower):
                found["temporal"] = True

            # Check for secondary symptom (after "and")
            if "and" in lower:
                after_and = lower.split("and", 1)[1]
                all_symptoms = specialties[spec]["primary_symptoms"] + specialties[spec].get("overlapping_symptoms", []) + specialties[spec].get("layman_terms", [])
                if any(symptom.lower() in after_and for symptom in all_symptoms):
                    found["secondary_symptom"] = True

            # Check for procedure mention
            if re.search(r"\bi had a (normal|inconclusive|abnormal|clear) .+ (last year|recently|a few months ago|in the past)\b", lower):
                found["procedure"] = True

            # Check for age and gender
            if re.search(r"i'm a \d{2,3}-year-old (male|female)", lower):
                found["age_gender"] = True

            # Check for medical history
            if "i have a history of" in lower:
                found["medical_history"] = True

            # Exit early if all parts found
            if all(found.values()):
                break

        # Print which parts were found (optional)
        print(f"{spec}: {found}")

        # At least some optional content should appear in 5 tries
        assert any(found.values()), f"No optional components found in complaints for specialty: {spec}"





