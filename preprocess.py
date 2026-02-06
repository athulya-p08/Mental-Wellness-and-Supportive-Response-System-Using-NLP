import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

nltk.download('punkt')
nltk.download('stopwords')

csv_path = "data/mental_health_data.csv"
df = pd.read_csv(csv_path)

# Remove extra whitespace in emotions
df['emotion'] = df['emotion'].str.strip()

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    return ' '.join(tokens)

df['cleaned_text'] = df['text'].apply(clean_text)

df = df[['text', 'cleaned_text', 'emotion']]

df.to_csv("processed_mental_health_data.csv", index=False, columns=['text','cleaned_text','emotion'])
print(df.head())
