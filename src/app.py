import torch
from src.utils.preprocessor import prepare_data
from src.models.multitask_model import MultiTargetAttentionModel
from src.training.trainer import train_and_evaluate
from src.training.predictor import predict_medical_complaint

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import joblib

# Configuration
BATCH_SIZE = 32
N_EPOCHS = 20
EARLY_STOPPING = 5
LEARNING_RATE = 0.001
MODEL_PATH = 'medical_complaint_model.pt'

# Load your data
df = pd.read_csv(r'C:\Users\Alka\Documents\Projects_torun\DiagnoSmart\Dataset.csv')  # replace with your actual CSV file

# Preprocess data
X_train, X_test, y_train, y_test, tfidf, specialty_encoder, severity_encoder = prepare_data(df)

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize model
model = MultiTargetAttentionModel(
    input_dim=X_train.shape[1],
    num_specialties=len(specialty_encoder.classes_),
    num_severities=len(severity_encoder.classes_),
    device=device
).to(device)

# Train and evaluate
train_and_evaluate(
    model=model,
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    batch_size=BATCH_SIZE,
    lr=LEARNING_RATE,
    n_epochs=N_EPOCHS,
    early_stopping_patience=EARLY_STOPPING,
    model_path=MODEL_PATH,
    device=device
)

# Load model for prediction
model.load_state_dict(torch.load(MODEL_PATH))

# Save encoders and vectorizer for later use
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
joblib.dump(specialty_encoder, 'specialty_encoder.pkl')
joblib.dump(severity_encoder, 'severity_encoder.pkl')

# Example prediction
sample_text = "I've been experiencing severe headaches and blurred vision for the past week."
result = predict_medical_complaint(
    complaint_text=sample_text,
    model=model,
    tfidf=tfidf,
    specialty_encoder=specialty_encoder,
    severity_encoder=severity_encoder,
    device=device
)

print("\nPrediction Example:")
print(result)
