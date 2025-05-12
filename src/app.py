import torch
from src.utils.preprocessor import prepare_data
from src.models.multitask_model import MultiTargetAttentionModel
from src.training.trainer import train_and_evaluate

from sklearn.model_selection import train_test_split
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
MODEL_PATH = 'medical_complaint_model.pt'

# Load dataset
df = pd.read_csv('DiagnoSmart/Dataset.csv')

# Preprocess data
X_train, X_test, y_train, y_test, tfidf, specialty_encoder, severity_encoder = prepare_data(df)

# Create DataLoader
train_loader = DataLoader(
    TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train[0]), torch.tensor(y_train[1]), torch.tensor(y_train[2])),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test[0]), torch.tensor(y_test[1]), torch.tensor(y_test[2])),
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

# Train and evaluate
train_and_evaluate(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    loss_fns=loss_fns,
    n_epochs=N_EPOCHS,
    early_stopping_patience=EARLY_STOPPING,
    model_path=MODEL_PATH,
    device=device
)

# Save encoders and vectorizer
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
joblib.dump(specialty_encoder, 'specialty_encoder.pkl')
joblib.dump(severity_encoder, 'severity_encoder.pkl')
