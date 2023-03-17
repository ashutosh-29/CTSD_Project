import pandas as pd
import string
import re
import nltk
from nltk.corpus import stopwords
from transliterate import translit, detect_language

nltk.download('stopwords')

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Detect the language of the text
    lang = detect_language(text)
    
    # Transliterate the text to Hindi if it is not already in Hindi
    if lang != 'hi':
        text = translit(text, 'hi')
    
    # Remove digits
    text = re.sub(r'\d+', '', text)
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stopwords.words('hindi')])
    
    return text

data = pd.read_csv('hindi_text.csv')

# Apply preprocessing to the text column
data['text'] = data['text'].apply(preprocess_text)

# Save preprocessed data to a new CSV file
data.to_csv('preprocessed_hindi_text.csv', index=False)
