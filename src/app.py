
import torch
import sys, os
import mlflow
import mlflow.pytorch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.preprocessor import prepare_data
from src.models.multitask_model import MultiTargetAttentionModel
from src.training.trainer import train_and_evaluate

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import joblib
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn

# Configuration
BATCH_SIZE = 32
N_EPOCHS = 20
EARLY_STOPPING = 5
LEARNING_RATE = 0.001
MODEL_PATH = r'C:\\Users\Alka\Documents\\Projects_torun\DiagnoSmart\saved_models\Diagnosmart_model.pt'

# Set MLflow Tracking URI and Experiment
mlflow.set_tracking_uri("./mlruns")
mlflow.set_experiment("DiagnoSmart")
experiment = mlflow.get_experiment_by_name("Diagnosmart")
print(f"Experiment ID: {experiment.experiment_id}" if experiment else "Experiment not found")

# Load dataset
df = pd.read_csv(r'C:\Users\Alka\Documents\Projects_torun\DiagnoSmart/Dataset.csv')

# Preprocess data
X_train, X_test, y_train, y_test, tfidf, specialty_encoder, severity_encoder = prepare_data(df)

# Assuming y_train is a NumPy array with three columns: [Specialty, Severity, Chronicity]
y_train_spec = torch.tensor(y_train[:, 0]).long()
y_train_sev = torch.tensor(y_train[:, 1]).long()
y_train_chr = torch.tensor(y_train[:, 2]).float()

y_test_spec = torch.tensor(y_test[:, 0]).long()
y_test_sev = torch.tensor(y_test[:, 1]).long()
y_test_chr = torch.tensor(y_test[:, 2]).float()

# Create DataLoader
train_loader = DataLoader(
    TensorDataset(torch.tensor(X_train).float(), y_train_spec, y_train_sev, y_train_chr),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    TensorDataset(torch.tensor(X_test).float(), y_test_spec, y_test_sev, y_test_chr),
    batch_size=BATCH_SIZE,
    shuffle=False
)

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

# Loss functions and optimizer
spec_loss_fn = nn.CrossEntropyLoss()
sev_loss_fn = nn.CrossEntropyLoss()
chr_loss_fn = nn.BCELoss()
loss_fns = (spec_loss_fn, sev_loss_fn, chr_loss_fn)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Start MLflow Run
with mlflow.start_run() as run:
    run_id = run.info.run_id

    # Log Model Parameters
    mlflow.log_param("learning_rate", LEARNING_RATE)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("n_epochs", N_EPOCHS)
    mlflow.log_param("early_stopping", EARLY_STOPPING)

    # Train and evaluate with MLflow logging
    train_and_evaluate(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fns=loss_fns,
        n_epochs=N_EPOCHS,
        early_stopping_patience=EARLY_STOPPING,
        model_path=MODEL_PATH,
        device=device,
        run_id=run_id 
    )

    # Save Encoders and Vectorizer
    save_path = r"C:\Users\Alka\Documents\Projects_torun\DiagnoSmart\saved_models"
    joblib.dump(tfidf, f'{save_path}\\tfidf_vectorizer.pkl')
    joblib.dump(specialty_encoder, f'{save_path}\\specialty_encoder.pkl')
    joblib.dump(severity_encoder, f'{save_path}\\severity_encoder.pkl')

    # Log Encoders as Artifacts
    mlflow.log_artifact('tfidf_vectorizer.pkl')
    mlflow.log_artifact('specialty_encoder.pkl')
    mlflow.log_artifact('severity_encoder.pkl')
