from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np

def prepare_tfidf(texts):
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 3), min_df=3)
    return tfidf.fit_transform(texts).toarray(), tfidf

def encode_labels(df, col):
    le = LabelEncoder()
    return le.fit_transform(df[col]), le

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np

def prepare_tfidf():
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 3), min_df=3)
    return tfidf

def encode_labels(df, col):
    le = LabelEncoder()
    return le.fit_transform(df[col]), le

def prepare_data(df):
    # Extract text and labels
    texts = df['complaint'].astype(str)  
    # specialties = df['specialty'] 
    # severities = df['severity_level']    
    # chronicities = df['chronic']  

    # Encode targets
    y_specialty, specialty_encoder = encode_labels(df, 'specialty')
    y_severity, severity_encoder = encode_labels(df, 'severity_level')
    y_chronicity = df['chronic'].astype(float).values  
    
    # Combine targets
    y = np.vstack((y_specialty, y_severity, y_chronicity)).T
    
     # Check for stratify eligibility
    unique_classes, class_counts = np.unique(y_specialty, return_counts=True)
    if np.any(class_counts < 2):
        # If any class has less than 2 samples, avoid stratifying
        stratify = None
    else:
        stratify = y_specialty
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, y, test_size=0.2, random_state=42, stratify=stratify
    )
    # TF-IDF vectorization
    tfidf = prepare_tfidf()
    
    X_train_transformed = tfidf.fit_transform(X_train).toarray()
    X_test_transformed  = tfidf.transform(X_test).toarray()

    return X_train_transformed, X_test_transformed, y_train, y_test, tfidf, specialty_encoder, severity_encoder

