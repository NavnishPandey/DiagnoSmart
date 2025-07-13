import torch
import numpy as np

from sentence_transformers import SentenceTransformer

def predict_medical_complaint(text, model, embedding_model, specialty_enc, severity_enc, device):
    model.eval()

    # Preprocess and encode
    embedding = embedding_model.encode([text])
    X_new = torch.FloatTensor(embedding).to(device)

    with torch.no_grad():
        spec_out, sev_out, chr_out = model(X_new)
        spec_label = specialty_enc.inverse_transform([np.argmax(spec_out.cpu().numpy())])[0]
        sev_label = severity_enc.inverse_transform([np.argmax(sev_out.cpu().numpy())])[0]
        chr_score = chr_out.cpu().numpy()[0][0]
        chr_label = "Chronic" if chr_score > 0.5 else "Acute"

    return {
        "specialty": spec_label,
        "severity_level": sev_label,
        "chronic_status": chr_label,
        "specialty_confidence": float(spec_out.max().item()),
        "severity_confidence": float(sev_out.max().item()),
        "chronic_confidence": float(chr_score) if chr_score > 0.5 else float(1 - chr_score)
    }


from sklearn.metrics import accuracy_score, roc_auc_score

def evaluate_test(model, test_loader, device):
    model.eval()
    all_spec_preds = []
    all_sev_preds = []
    all_chr_preds = []

    all_spec_labels = []
    all_sev_labels = []
    all_chr_labels = []

    with torch.no_grad():
        for X_batch, spec, sev, chr in test_loader:
            X_batch = X_batch.to(device)
            spec_pred, sev_pred, chr_pred = model(X_batch)

            # Specialty and Severity: multi-class classification → take argmax
            spec_pred_labels = torch.argmax(spec_pred, dim=1).cpu().numpy()
            sev_pred_labels = torch.argmax(sev_pred, dim=1).cpu().numpy()

            # Chronicity: output probabilities, threshold 0.5 to binarize
            chr_pred_scores = chr_pred.cpu().numpy().flatten()
            chr_pred_labels = (chr_pred_scores > 0.5).astype(int)

            # Accumulate predictions and labels
            all_spec_preds.extend(spec_pred_labels)
            all_sev_preds.extend(sev_pred_labels)
            all_chr_preds.extend(chr_pred_labels)

            all_spec_labels.extend(spec.cpu().numpy())
            all_sev_labels.extend(sev.cpu().numpy())
            all_chr_labels.extend(chr.cpu().numpy().astype(int))  # assume 0/1 float

    # Calculate metrics
    spec_acc = accuracy_score(all_spec_labels, all_spec_preds)
    sev_acc = accuracy_score(all_sev_labels, all_sev_preds)

    # For chronicity calculate accuracy and AUC (if possible)
    chr_acc = accuracy_score(all_chr_labels, all_chr_preds)

    try:
        chr_auc = roc_auc_score(all_chr_labels, all_chr_preds)
    except ValueError:
        # Case where only one class is present
        chr_auc = None

    metrics = {
        "test_specialty_accuracy": spec_acc,
        "test_severity_accuracy": sev_acc,
        "test_chronicity_accuracy": chr_acc,
    }

    if chr_auc is not None:
        metrics["test_chronicity_auc"] = chr_auc

    return metrics
