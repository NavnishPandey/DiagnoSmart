from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
import joblib
import numpy as np
import os, sys

# Set path to find model and utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.multitask_model import MultiTargetAttentionModel
from src.training.predictor import predict_medical_complaint

# Flask setup
app = Flask(__name__)
CORS(app)

# File paths

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir)) # C:/.../ (one level up)

MODEL_PATH = os.path.join(ROOT_DIR, "saved_models", "medical_complaint_model.pt")
EMBENDING_PATH = os.path.join(ROOT_DIR, "saved_models", "embending_model.pkl")
SPECIALTY_ENCODER_PATH = os.path.join(ROOT_DIR, "saved_models", "specialty_encoder.pkl")
SEVERITY_ENCODER_PATH = os.path.join(ROOT_DIR, "saved_models", "severity_encoder.pkl")
# MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "Diagnosmart_model.pt")
# TFIDF_PATH = os.path.join(BASE_DIR, "saved_models", "tfidf_vectorizer.pkl")
# SPECIALTY_ENCODER_PATH = os.path.join(BASE_DIR, "saved_models", "specialty_encoder.pkl")
# SEVERITY_ENCODER_PATH = os.path.join(BASE_DIR, "saved_models", "severity_encoder.pkl")


# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# Load vectorizer and encoders
sentence_transformer = joblib.load(EMBENDING_PATH)
specialty_enc = joblib.load(SPECIALTY_ENCODER_PATH)
severity_enc = joblib.load(SEVERITY_ENCODER_PATH)

# Get model input dimensions
input_dim = len(sentence_transformer.get_feature_names_out())
num_specialties = len(specialty_enc.classes_)
num_severities = len(severity_enc.classes_)

# Load model
model = MultiTargetAttentionModel(
    input_dim=input_dim,
    num_specialties=num_specialties,
    num_severities=num_severities,
    device=device
).to(device)

state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.load_state_dict(state_dict)
model.eval()
@app.route('/')
def home():
    return render_template('index.html')  # Optional frontend

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'complaint' not in data:
        return jsonify({'error': 'No complaint provided'}), 400

    complaint = data['complaint']
    print(f"Received complaint: {complaint}")

    try:
        prediction = predict_medical_complaint(
            complaint, model, sentence_transformer, specialty_enc, severity_enc, device
        )
        print(prediction)
        return jsonify(prediction)
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
