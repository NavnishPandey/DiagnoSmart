specialties = {
    "Dermatology": {
        "common_complaints": [
            "I have a rash that won't go away",
            "My skin is very itchy and red",
            "I noticed an unusual mole on my back",
            "My acne is getting worse despite over-the-counter treatments",
            "I have patches of dry, scaly skin",
            "My scalp is very itchy and has flakes",
            "I have blisters on my hands",
            "There's a growth on my skin that's changing color",
            "I'm losing hair in patches",
            "I have painful sores around my mouth"
        ],
        "primary_symptoms": ["rash", "itching", "mole", "acne", "skin lesion", "discoloration",
                    "dryness", "scaling", "hair loss", "hives", "warts", "skin growth",
                    "eczema", "dermatitis", "psoriasis", "skin infection"],
        "overlapping_symptoms": ["fatigue", "pain", "swelling", "redness", "fever", "nausea"],
        "procedures": ["biopsy", "patch test", "cryotherapy", "topical steroids", "phototherapy"],
        "layman_terms": ["itchy skin", "breakout", "pimple", "dry patches", "flaky scalp", "blister"]
    },
    "Cardiology": {
        "common_complaints": [
            "I feel pain in my chest when I exercise",
            "My heart races even when I'm resting",
            "I get short of breath climbing stairs",
            "My ankles are swollen at the end of the day",
            "I wake up at night feeling like I can't breathe",
            "I sometimes feel lightheaded and like I might pass out",
            "My blood pressure readings have been very high",
            "I can feel my heart skipping beats"
        ],
        "primary_symptoms": ["chest pain", "palpitations", "shortness of breath", "edema",
                    "hypertension", "arrhythmia", "heart murmur", "cyanosis", "claudication", "orthopnea"],
        "overlapping_symptoms": ["dizziness", "syncope", "fatigue", "nausea", "sweating", "anxiety", "weakness"],
        "procedures": ["ECG", "stress test", "echocardiogram", "angiogram", "Holter monitor", "cardiac catheterization"],
        "layman_terms": ["heart racing", "skipped beats", "tightness in chest", "winded", "puffy feet", "blacked out"]
    },
    "Neurology": {
        "primary_symptoms": [
            "numbness", "tingling", "muscle weakness", "vision changes",
            "difficulty speaking", "slurred speech", "confusion",
            "seizures", "tremors", "poor coordination", "dizziness", "balance issues",
            "memory problems", "headache", "migraine"
        ],
        "overlapping_symptoms": ["fatigue", "sleep disturbance", "blurred vision"],
        "common_complaints": [
            "I often feel numbness in my hands and feet",
            "I've been having frequent migraines and memory lapses",
            "I experience muscle weakness and tingling in my limbs",
            "I've had trouble with coordination and dizziness"
        ],
        "layman_terms": ["shaky hands", "blackouts", "forgetfulness", "blurry vision"],
        "procedures": ["EEG", "MRI", "CT scan", "nerve conduction study"]
    },
    "Gastroenterology": {
        "common_complaints": [
            "I have persistent heartburn that won't go away with antacids",
            "I've been having diarrhea for over a week",
            "I notice blood in my stool",
            "I have severe pain in my abdomen after eating",
            "I feel nauseous most mornings",
            "I've lost weight without trying",
            "I've been constipated for days despite trying laxatives",
            "My skin and eyes look yellow",
            "I have difficulty swallowing food"
        ],
        "primary_symptoms": ["abdominal pain", "heartburn", "diarrhea", "constipation",
                    "blood in stool", "jaundice", "difficulty swallowing", "bloating",
                    "indigestion", "reflux", "hemorrhoids", "rectal bleeding", "dyspepsia"],
        "overlapping_symptoms": ["nausea", "vomiting", "weight loss", "fatigue", "pain", "fever", "appetite changes"],
        "procedures": ["endoscopy", "colonoscopy", "ultrasound", "HIDA scan", "barium swallow", "pH monitoring"],
        "layman_terms": ["upset stomach", "the runs", "can't keep food down", "blocked up", "heartburn", "throwing up blood"]
    },
    "Ophthalmology": {
        "common_complaints": [
            "My vision is getting blurry",
            "I see halos around lights at night",
            "My eyes are constantly dry and irritated",
            "I have pain in my eye when looking at bright light",
            "My eye is red and feels like there's something in it",
            "I've been seeing floating spots in my vision",
            "My eyelid has been twitching for days",
            "I wake up with crusty, sticky eyes",
            "I have double vision sometimes"
        ],
        "primary_symptoms": ["blurry vision", "eye pain", "redness", "dryness", "floaters",
                    "double vision", "light sensitivity", "vision loss", "tearing",
                    "night blindness", "eye discharge", "foreign body sensation"],
        "overlapping_symptoms": ["headache", "dizziness", "nausea", "fatigue", "itching"],
        "procedures": ["visual field test", "tonometry", "dilated eye exam", "slit lamp exam", "retinal imaging"],
        "layman_terms": ["fuzzy vision", "see spots", "gritty eyes", "watery eyes", "can't see at night", "cross-eyed"]
    },
    "Urology": {
        "common_complaints": [
            "I have a burning sensation when urinating",
            "I need to urinate frequently, even at night",
            "I see blood in my urine",
            "I have pain in my lower back near my kidneys",
            "I have difficulty starting or maintaining a urine stream",
            "I feel like my bladder is never completely empty",
            "I have pain during intercourse",
            "I leak urine when I cough or sneeze",
            "I noticed a lump in my testicle"
        ],
        "primary_symptoms": ["dysuria", "hematuria", "frequency", "urgency", "incontinence",
                    "kidney pain", "hesitancy", "nocturia", "urinary retention",
                    "testicular pain", "erectile dysfunction", "urethral discharge"],
        "overlapping_symptoms": ["pain", "fever", "nausea", "fatigue", "abdominal pain", "back pain"],
        "procedures": ["cystoscopy", "urodynamic testing", "PSA test", "urinalysis", "ultrasound", "prostate exam"],
        "layman_terms": ["peeing hurts", "can't hold it", "blood in pee", "weak stream", "accidents", "leaking"]
    },

    "Endocrinology": {
        "common_complaints": [
            "I'm always thirsty and urinating frequently",
            "I've gained weight despite eating normally",
            "I feel tired all the time and my hair is thinning",
            "My hands shake and I feel anxious for no reason",
            "I've noticed a lump in my neck below my Adam's apple",
            "My skin has darkened in patches",
            "I get dizzy if I don't eat every few hours",
            "My periods have become irregular"
        ],
        "primary_symptoms": ["fatigue", "weight changes", "excessive thirst", "polyuria",
                            "heat intolerance", "cold intolerance", "hair thinning",
                            "tremors", "neck swelling", "skin darkening"],
        "overlapping_symptoms": ["anxiety", "palpitations", "nausea", "dizziness", "weakness"],
        "procedures": ["thyroid ultrasound", "HbA1c test", "glucose tolerance test",
                      "thyroid function tests", "cortisol test"],
        "layman_terms": ["sugar problems", "thyroid issue", "hormone imbalance",
                        "always hungry", "crashing energy"]
    },
    "Pulmonology": {
        "common_complaints": [
            "I can't stop coughing, especially at night",
            "I wheeze when I breathe out",
            "I get short of breath just walking around the house",
            "I cough up thick yellow mucus every morning",
            "I wake up gasping for air",
            "My chest feels tight when I'm around dust",
            "I've had a cough for over 3 months",
            "I feel like I can't take a deep breath"
        ],
        "primary_symptoms": ["cough", "wheezing", "shortness of breath", "sputum production",
                           "chest tightness", "hemoptysis", "apnea", "hypoxia"],
        "overlapping_symptoms": ["fatigue", "fever", "weight loss", "nasal congestion", "headache"],
        "procedures": ["chest X-ray", "pulmonary function test", "bronchoscopy",
                      "CT scan of chest", "arterial blood gas"],
        "layman_terms": ["can't catch breath", "whistling chest", "smoker's cough",
                        "phlegm", "air hunger"]
    },
    "Orthopedics": {
        "common_complaints": [
            "My knee locks up when I try to stand",
            "I have sharp pain in my shoulder when I reach overhead",
            "My back goes out if I bend the wrong way",
            "I heard a pop in my ankle and now it's swollen",
            "My fingers get stiff and painful in the morning",
            "I have numbness that shoots down my leg",
            "My wrist hurts when I grip things",
            "My hip makes a clicking sound when I walk"
        ],
        "primary_symptoms": ["joint pain", "swelling", "stiffness", "limited range of motion",
                           "instability", "crepitus", "radiating pain", "muscle weakness"],
        "overlapping_symptoms": ["numbness", "tingling", "fatigue", "headache", "sleep disturbances"],
        "procedures": ["X-ray", "MRI", "arthroscopy", "joint injection", "bone scan"],
        "layman_terms": ["throwing out back", "trick knee", "tennis elbow",
                         "pinched nerve", "bone-on-bone"]
    },
    "Rheumatology": {
        "common_complaints": [
            "All my joints ache when I wake up",
            "My fingers are swollen like sausages",
            "I'm so stiff in the morning it takes hours to loosen up",
            "I have red, scaly patches on my elbows and knees",
            "Even my jaw hurts when I chew",
            "My symptoms get worse when it's cold or rainy",
            "I feel exhausted no matter how much I sleep",
            "My fingers turn white when I'm cold"
        ],
        "primary_symptoms": ["joint pain", "morning stiffness", "swelling", "skin rashes",
                           "fatigue", "Raynaud's phenomenon", "dry eyes/mouth", "fever"],
        "overlapping_symptoms": ["depression", "numbness", "headache", "weight loss", "muscle pain"],
        "procedures": ["ANA test", "rheumatoid factor", "joint aspiration",
                      "anti-CCP test", "ESR/CRP tests"],
        "layman_terms": ["all-over pain", "weather pain", "flare-up",
                         "autoimmune issues", "body attacking itself"]
    },
    "ENT (Otolaryngology)": {
        "common_complaints": [
            "I keep losing my voice",
            "My ears feel plugged all the time",
            "I have recurring sinus infections",
            "There's a constant ringing in my ears",
            "I wake up choking because of acid in my throat",
            "I snore so loud it wakes me up",
            "I've had a sore throat for over a month",
            "I feel dizzy when I turn my head"
        ],
        "primary_symptoms": ["hearing loss", "tinnitus", "vertigo", "hoarseness",
                           "nasal obstruction", "dysphagia", "facial pain", "neck mass"],
        "overlapping_symptoms": ["headache", "cough", "fatigue", "nausea", "fever"],
        "procedures": ["audiogram", "nasal endoscopy", "CT sinuses",
                       "laryngoscopy", "VNG testing"],
        "layman_terms": ["clogged ears", "lump in throat", "post-nasal drip",
                         "whistling nose", "room spinning"]
    },
    "Psychiatry": {
        "common_complaints": [
            "I can't shut off my thoughts at night",
            "I've lost interest in everything I used to enjoy",
            "I panic in crowds and can't leave my house",
            "I see shadows that aren't really there",
            "I go days without sleeping but have endless energy",
            "I wash my hands until they bleed",
            "I have flashbacks to my time in the war",
            "I hear voices when no one is around"
        ],
        "primary_symptoms": ["anxiety", "depressed mood", "hallucinations", "insomnia",
                           "mania", "intrusive thoughts", "panic attacks", "dissociation"],
        "overlapping_symptoms": ["fatigue", "headache", "nausea", "weight changes", "palpitations"],
        "procedures": ["psych evaluation", "cognitive testing", "sleep study",
                      "MMPI", "depression screening"],
        "layman_terms": ["nervous breakdown", "lost my spark", "racing thoughts",
                         "hearing things", "can't cope"]
    },
    "Hematology": {
        "primary_symptoms": [
            "anemia", "easy bruising", "bleeding", "petechiae", "fatigue",
            "paleness", "clotting issues", "swollen lymph nodes"
        ],
        "overlapping_symptoms": [
            "weakness", "dizziness", "shortness of breath", "fever", "night sweats"
        ],
        "procedures": [
            "CBC", "bone marrow biopsy", "coagulation panel", "iron panel", "peripheral smear"
        ],
        "common_complaints": [
            "I bruise very easily and don't know why",
            "I'm constantly tired and pale",
            "My gums bleed when I brush my teeth",
            "I feel dizzy and short of breath with minimal activity",
            "I have frequent nosebleeds",
            "I feel cold all the time",
            "I get infections easily"
        ],
        "layman_terms": [
            "always tired", "bleed easily", "pale skin", "low blood", "random bruises",
            "blood problems", "blood not clotting", "easy bleeding"
        ]
    },
    "Oncology": {
        "primary_symptoms": [
            "unintentional weight loss", "fatigue", "mass", "lymphadenopathy",
            "pain", "night sweats", "bleeding", "persistent cough"
        ],
        "overlapping_symptoms": [
            "fever", "appetite loss", "shortness of breath", "weakness"
        ],
        "procedures": [
            "CT scan", "MRI", "biopsy", "tumor markers", "PET scan"
        ],
        "common_complaints": [
            "I found a lump that won’t go away",
            "I’ve lost a lot of weight without trying",
            "I have night sweats and I feel weak",
            "My fatigue is unbearable and it’s getting worse",
            "My pain doesn’t respond to regular meds"
        ],
        "layman_terms": [
            "lump", "something growing", "feels like cancer", "constant tiredness", "sick all the time"
        ]
    },
    "Immunology_Infectious": {
        "primary_symptoms": [
            "hives", "itching", "runny nose", "sneezing", "wheezing",
            "anaphylaxis", "eczema", "recurrent infections", "fever", "chills",
            "night sweats", "rash", "swollen lymph nodes", "diarrhea", "persistent cough"
        ],
        "overlapping_symptoms": [
            "fatigue", "nausea", "headache", "pain", "shortness of breath",
            "redness", "swelling"
        ],
        "procedures": [
            "allergy skin test", "IgE testing", "spirometry", "immunoglobulin levels",
            "food challenge test", "blood culture", "serologic testing", "chest X-ray", "infectious disease panel"
        ],
        "common_complaints": [
            "I break out in hives after I eat certain foods",
            "I have constant sneezing and runny nose",
            "My eyes itch terribly during spring",
            "I get swelling and shortness of breath after bee stings",
            "I have frequent infections",
            "I’ve had a fever that won’t go away",
            "I just got back from a trip and now I’m really sick",
            "My wound looks infected",
            "I’ve had diarrhea and vomiting for 5 days"
        ],
        "layman_terms": [
            "allergic reaction", "can't breathe after eating", "puffy face",
            "seasonal allergies", "itchy eyes", "won’t stop being sick",
            "caught something", "weird infection", "travel bug", "bad fever"
        ]
    },
    "Emergency Medicine": {
        "common_complaints": [
            "I suffered a head trauma and now I'm confused and bleeding",
            "I fell from a ladder and can't move my leg, it's severely swollen",
            "My child collapsed suddenly and is unresponsive",
            "I have a deep laceration on my forearm with heavy bleeding",
            "After a car accident, I have chest pain and trouble breathing",
            "I'm burning up with a high fever and can't stop shaking, went to the ER"
        ],
        "primary_symptoms": [
            "severe trauma", "blunt force injury", "third-degree burns", "acute chest pain",
            "acute abdominal pain", "sudden loss of consciousness",
            "drug overdose", "severe dyspnea", "shock"
        ],
        "overlapping_symptoms": [
            "intense pain", "dizziness with trauma", "confusion after injury", 
            "persistent nausea post-impact", "high fever with chills", 
            "uncontrolled bleeding", "sudden weakness", "loss of motor control",
            "shortness of breath after accident", "irregular heartbeat"
        ],
        "procedures": [
            "trauma protocol", "whole-body CT scan", "emergency resuscitation",
            "ER complete blood panel", "fracture stabilization", 
            "emergency IV fluids and oxygen therapy"
        ],
        "layman_terms": [
            "I smashed my head and now I’m bleeding", "I fell hard and can’t move my leg", 
            "He just collapsed and won’t wake up", "cut won’t stop bleeding", 
            "after the crash, my chest hurts and I can’t breathe", 
            "I overdosed and feel faint", "burned badly cooking", 
            "took a bad fall and blacked out", "can’t feel my legs after the accident",
            "threw up and passed out from the fever"
        ]

    },
    "General_medicine": {
        "primary_symptoms": [
            "general discomfort", "fatigue", "lightheadedness", "low-grade fever", "vague headache",
            "mild nausea", "loss of appetite", "non-specific pain", "night sweats", "paleness"
        ],
        "common_complaints": [
            "I don't feel well", "I feel off", "I'm sick", "Something's wrong but I don't know what",
            "I can't explain what's happening", "I've been feeling worse lately",
            "It's probably nothing but I'm worried", "I hurt all over", "I have random aches",
            "I don't feel like myself", "I feel weak and tired", "I have a low fever",
            "I feel dizzy and lightheaded", "I have a headache that won't go away",
            "I feel nauseous but I don't know why", "I haven't been eating well",
            "I have a cough", "I have chills"
        ],
        "procedures": [
            "general check-up", "routine bloodwork", "annual physical",
            "physical exam", "initial evaluation", "medical advice", "blood pressure check"
        ],
        "overlapping_symptoms": [
            "headache", "back pain", "generic pain", "fever", "dizziness",
            "leg pain", "muscle aches", "tightness", "nausea", "anxiety"
        ],
        "layman_terms": [
            "I feel sick", "I have a cold", "I think I'm coming down with something",
            "I have a headache", "I feel weak", "I have a stomach ache",
            "I feel feverish", "I have a sore throat", "I feel dizzy",
            "I have body aches"
        ]
    }
}
# Severity indicators
severity_indicators = {
    "low": ["mild", "slight", "minor", "a bit of", "somewhat"],
    "moderate": ["moderate", "noticeable", "uncomfortable", "bothersome", "concerning"],
    "high": ["severe", "intense", "extreme", "debilitating", "excruciating", "unbearable"]
}

# Temporal patterns
temporal_patterns = {
    "constant": ["constant", "continuous", "persistent", "ongoing", "unrelenting", "steady"],
    "intermittent": ["intermittent", "comes and goes", "periodic", "occasional", "sporadic", "fluctuating"],
    "progressive": ["worsening", "gradually increasing", "getting worse", "progressive", "escalating"],
    "cyclical": ["cyclical", "recurring", "in cycles", "comes in waves", "periodic", "episodic"],
    "acute": ["sudden onset", "acute", "abrupt", "came on suddenly", "all of a sudden"]
}

positive_modifiers = ["sometimes", "mostly", "occasional", "intermittent", "frequent", "persistent"]
negative_modifiers = ["no", "not sure if", "no history of", "never had"]

chronic_indicators = {
    True: ["for years", "since childhood", "chronic", "long-standing", "for as long as I can remember",
           "for the past several years", "ongoing for over a year","almost a year","several months", "about six months"],
    False: ["recent", "just started", "new", "started last week", "began recently",
            "for the first time", "never happened before",  "a few days", "about a week", "two weeks", "several weeks",
    "a month", "the past couple days", "since last Tuesday", "on and off for weeks"]
}

compatible_temporal_for_chronic = {
    True: ["constant", "intermittent", "progressive", "cyclical"],
    False: ["acute", "intermittent", "cyclical", "progressive"]
}

import pandas as pd
import numpy as np
from faker import Faker
import random

# Reproducibility
random.seed(42)
np.random.seed(42)
fake = Faker()
Faker.seed(42)


def introduce_typos(text, prob=0.1):
    if not isinstance(text, str):
        raise TypeError("Input 'text' must be a string.")
    if not (0.0 <= prob <= 1.0):
        raise ValueError("Probability must be between 0.0 and 1.0")
    words = text.split()
    for i in range(len(words)):
        if random.random() < prob and len(words[i]) > 3:
            pos = random.randint(1, len(words[i]) - 2)
            words[i] = words[i][:pos] + words[i][pos] + words[i][pos:]
    return ' '.join(words)

def generate_complaint(specialty):
    specialty_data = specialties[specialty]
    severity_level = random.choice(["low", "moderate", "high"])
    is_chronic = random.choice([True, False])

    # Demographics
    include_demographics = random.random() < 0.3
    if include_demographics:
        age = random.randint(18, 85)
        gender = random.choice(["male", "female"])
        demographics = f"I'm a {age}-year-old {gender}."
    else:
        demographics = ""

    # Main symptom or complaint
    if random.random() < 0.6 and specialty_data["common_complaints"]:
        main_symptom = random.choice(specialty_data["common_complaints"])
    else:
        symptom_pool = specialty_data["primary_symptoms"] + specialty_data.get("overlapping_symptoms", [])
        main_symptom = random.choice(symptom_pool)

    severity = random.choice(severity_indicators[severity_level])
    main_sentence = f"I have {severity} {main_symptom}"

    # Temporal expression
    if random.random() < 0.6:
        temporal_type = random.choice(compatible_temporal_for_chronic[is_chronic])
        temporal = random.choice(temporal_patterns[temporal_type])
        time_period = random.choice(chronic_indicators[is_chronic])
        temporal_sentence = f"It has been {temporal} for {time_period}."
    else:
        temporal_sentence = ""

    # Secondary symptom
    if random.random() < 0.7:
        second_symptom_pool = specialty_data["primary_symptoms"] + specialty_data["overlapping_symptoms"]
        second_symptom = random.choice(second_symptom_pool)
        if random.random() < 0.3 and specialty_data["layman_terms"]:
            second_symptom = random.choice(specialty_data["layman_terms"])
        modifier = random.choice(positive_modifiers) if random.random() < 0.8 else random.choice(negative_modifiers)
        connector = "and" if modifier in positive_modifiers else "but"
        secondary_sentence = f"{connector.capitalize()} it's {modifier} {second_symptom}."
    else:
        secondary_sentence = ""

    # Procedure
    if random.random() < 0.4 and specialty_data["procedures"]:
        procedure = random.choice(specialty_data["procedures"])
        time_ref = random.choice(["last year", "recently", "a few months ago", "in the past"])
        result = random.choice(["normal", "inconclusive", "abnormal", "clear"])
        procedure_sentence = f"I had a {result} {procedure} {time_ref}."
    else:
        procedure_sentence = ""

    # Medical history
    if random.random() < 0.25:
        condition = random.choice(["diabetes", "high blood pressure", "asthma", "thyroid disorder"])
        history_sentence = f"I have a history of {condition}."
    else:
        history_sentence = ""

    # Combine all parts
    full_text = " ".join(filter(None, [
        demographics,
        main_sentence + ".",
        temporal_sentence,
        secondary_sentence,
        procedure_sentence,
        history_sentence
    ])).strip()

    # Optional: Introduce typos
    if random.random() < 0.1:
        full_text = introduce_typos(full_text)

    return full_text, severity_level, is_chronic
import re

def light_postprocess(text):
    # 1. Remove repeated "I have"
    text = re.sub(r"\b(I have)\s+\1\b", r"\1", text)

    # 2. Fix awkward sequences like "unbearable I see" → "unbearable. I see"
    text = re.sub(r"(\b(?:severe|intense|unbearable|debilitating|noticeable|mild|sharp|chronic)\b)\s+(I\b)", r"\1. \2", text)

    # 3. Remove repeated prepositions like "for for years"
    text = re.sub(r"\bfor\s+for\b", "for", text)

    # 4. Capitalize first letter if missing
    if text and not text[0].isupper():
        text = text[0].upper() + text[1:]

    # 5. Remove excessive punctuation like "...."
    text = re.sub(r"\.\.+", ".", text)

    # 6. Clean up extra spaces
    text = re.sub(r"\s{2,}", " ", text)

    return text.strip()


def generate_synthetic_dataset(n_samples=1000):
    data = []
    specialties_list = list(specialties.keys())
    samples_per_specialty = n_samples // len(specialties_list)

    for specialty in specialties_list:
        for _ in range(samples_per_specialty):
            complaint, severity_level, is_chronic = generate_complaint(specialty)
            complaint = light_postprocess(complaint)
            data.append({
                "complaint": complaint,
                "specialty": specialty,
                "severity_level": severity_level,
                "chronic": is_chronic
            })

    random.shuffle(data)
    return pd.DataFrame(data)

# Generate dataset
df = generate_synthetic_dataset(4000)
df.to_csv('Dataset.csv')