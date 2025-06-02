import joblib
import os

# Path to your saved vectorizer
VECTORIZER_PATH = r"C:\Users\Alka\Documents\Projects_torun\DiagnoSmart\saved_models\tfidf_vectorizer.pkl"

# Load the saved TfidfVectorizer
def load_vectorizer():
    if not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(f"Vectorizer file not found at: {VECTORIZER_PATH}")
    
    vectorizer = joblib.load(VECTORIZER_PATH)
    return vectorizer

# Preprocess and transform the input complaint using the loaded vectorizer
def vectorize_complaint(complaint_text):
    vectorizer = load_vectorizer()
    
    # Basic cleaning (optional – depends on your model’s training)
    cleaned_text = complaint_text.strip().lower()
    
    # Transform to vector
    vector = vectorizer.transform([cleaned_text])
    return vector
