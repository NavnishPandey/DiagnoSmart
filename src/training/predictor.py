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
