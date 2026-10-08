import nltk
from nltk.stem import WordNetLemmatizer
import string

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """
    Preprocess text using NLP techniques:
    1. Lowercase
    2. Tokenize
    3. Remove punctuation
    4. Lemmatize
    """
    # Tokenize and lowercase
    tokens = nltk.word_tokenize(text.lower())
    
    # Remove punctuation and lemmatize
    clean_tokens = []
    for token in tokens:
        if token not in string.punctuation:
            clean_tokens.append(lemmatizer.lemmatize(token))
            
    return " ".join(clean_tokens)
