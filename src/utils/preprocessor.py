from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np

def prepare_tfidf(texts):
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 3), min_df=5)
    return tfidf.fit_transform(texts).toarray(), tfidf

def encode_labels(df, col):
    le = LabelEncoder()
    return le.fit_transform(df[col]), le

def prepare_data(df):
    # Extract text and labels
    texts = df['complaint'].astype(str)  
    specialties = df['specialty'] 
    severities = df['severity_level']    
    chronicities = df['chronic']  # Add this line

    # TF-IDF vectorization
    X, tfidf = prepare_tfidf(texts)

    # Encode targets
    y_specialty, specialty_encoder = encode_labels(df, 'specialty')
    y_severity, severity_encoder = encode_labels(df, 'severity_level')
    y_chronicity = df['chronic'].astype(float).values  # Assuming chronicity is already a float

    # Combine targets
    
    y = np.vstack((y_specialty, y_severity, y_chronicity)).T  # shape: (n_samples, 3)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y_specialty
    )

    return X_train, X_test, y_train, y_test, tfidf, specialty_encoder, severity_encoder
