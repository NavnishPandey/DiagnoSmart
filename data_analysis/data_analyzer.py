import pandas as pd
df = pd.read_csv(r"C:\Users\giova\OneDrive\Documenti\DiagnoSmart\Dataset.csv")  
print(df.columns)
df.head()
print(df['specialty'].value_counts())
from collections import Counter
import re

def get_common_words_for_specialty(specialty, top_k=20):
    texts = df[df['specialty'] == specialty]['complaint']
    all_words = []
    for text in texts:
        words = re.findall(r'\b\w+\b', text.lower())
        all_words.extend(words)
    return Counter(all_words).most_common(top_k)

get_common_words_for_specialty('Orthopedics')
get_common_words_for_specialty('Neurology')
df[df['complaint'].str.contains("injury|headache|pain", case=False)]

