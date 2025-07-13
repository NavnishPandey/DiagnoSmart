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

# Download the list of English stopwords from NLTK
nltk.download('stopwords')

# Load the small English model from spaCy for NLP tasks (tokenization, lemmatization, etc.)
nlp = spacy.load("en_core_web_sm")

# Create a set of English stopwords from NLTK's list for easy lookup during text preprocessing
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    # Convert the input text to lowercase to standardize it
    text = text.lower()
    
    # Remove all digits from the text using a regular expression
    text = re.sub(r'\d+', '', text)
    
    # Remove all characters that are NOT word characters or whitespace
    # This effectively removes punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    
    # Process the cleaned text with the spaCy NLP pipeline
    doc = nlp(text)
    
    # Lemmatize each token and filter out stopwords
    # Lemmatization reduces words to their base or dictionary form
    tokens = [token.lemma_ for token in doc if token.lemma_ not in stop_words]
    
    # Join the filtered lemmas back into a single string separated by spaces
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

    # Stratify only if every class in specialty has at least 3 samples (needed for 3 splits)
    _, class_counts = np.unique(y_specialty, return_counts=True)
    stratify = y_specialty if np.all(class_counts >= 3) else None

    # First split train (70%) and temp (30%)
    X_train_texts, X_temp_texts, y_train, y_temp = train_test_split(
        texts, y, test_size=0.3, random_state=42, stratify=stratify
    )

    # For temp (val + test), if stratify is None fallback to None (no stratification)
    stratify_temp = None
    if stratify is not None:
        # Extract specialty from y_temp for stratification in val/test split
        y_temp_specialty = y_temp[:, 0]
        _, counts_temp = np.unique(y_temp_specialty, return_counts=True)
        if np.all(counts_temp >= 2):
            stratify_temp = y_temp_specialty

    # Split temp into validation (15%) and test (15%)
    X_val_texts, X_test_texts, y_val, y_test = train_test_split(
        X_temp_texts, y_temp, test_size=0.5, random_state=42, stratify=stratify_temp
    )

    # Load SentenceTransformer model once here
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate embeddings for train, val, test
    X_train = model.encode(X_train_texts.tolist(), show_progress_bar=True)
    X_val = model.encode(X_val_texts.tolist(), show_progress_bar=True)
    X_test = model.encode(X_test_texts.tolist(), show_progress_bar=True)

    return X_train, X_val, X_test, y_train, y_val, y_test, model, specialty_encoder, severity_encoder



