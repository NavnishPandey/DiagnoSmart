from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
import numpy as np
import re
import spacy
import nltk
from nltk.corpus import stopwords

# def prepare_tfidf(texts):
#     #Crrate a TF-IDF vectorizer, dismiss words that appear in less than 3 documents
#     # and use unigrams, bigrams, and trigrams
#     # as features, limiting to a maximum of 5000 features 
#     tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 3), min_df=3)
#     #Use the vectorizer to transform the texts and return the resulting matrix and the vectorizer
#     return tfidf.fit_transform(texts).toarray(), tfidf

def prepare_embeddings(texts):
    model = SentenceTransformer('all-MiniLM-L6-v2')  
    embeddings = model.encode(texts.tolist(), show_progress_bar=True)
    return embeddings, model

nltk.download('stopwords')
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.lemma_ not in stop_words]
    return ' '.join(tokens)


# Function to encode labels using LabelEncoder
def encode_labels(df, col):
    le = LabelEncoder()
    # Fit and transform the specified column of the DataFrame
    # Return the transformed labels and the fitted LabelEncoder
    return le.fit_transform(df[col]), le

def prepare_data(df):
    # Preprocess the 'complaint' column
    texts = df['complaint'].astype(str).apply(preprocess_text)

    # Encode targets
    y_specialty, specialty_encoder = encode_labels(df, 'specialty')
    y_severity, severity_encoder = encode_labels(df, 'severity_level')
    y_chronicity = df['chronic'].astype(float).values

    # Combine targets
    y = np.vstack((y_specialty, y_severity, y_chronicity)).T

    # Stratified split (based on specialty, if possible)
    unique_classes, class_counts = np.unique(y_specialty, return_counts=True)
    stratify = y_specialty if np.all(class_counts >= 2) else None

    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        texts, y, test_size=0.2, random_state=42, stratify=stratify
    )

    # Compute embeddings only after split
    model = SentenceTransformer('all-MiniLM-L6-v2')
    X_train = model.encode(X_train_texts.tolist(), show_progress_bar=True)
    X_test = model.encode(X_test_texts.tolist(), show_progress_bar=True)

    return X_train, X_test, y_train, y_test, model, specialty_encoder, severity_encoder


