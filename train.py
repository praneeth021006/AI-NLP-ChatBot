import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from nlp_utils import preprocess_text

def train_model():
    print("Loading intent dataset...")
    with open('intents.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 1. Collect conversational training data
    patterns = []
    tags = []

    for intent in data['intents']:
        for pattern in intent['patterns']:
            patterns.append(pattern)
            tags.append(intent['tag'])

    # 2. Preprocess text data using NLP techniques
    print("Preprocessing text data...")
    processed_patterns = [preprocess_text(p) for p in patterns]

    # 3 & 4. Train the chatbot using scikit-learn (TF-IDF + Logistic Regression)
    print("Training intent classification model...")
    # Create a pipeline that vectorizes the text and then trains a classifier
    model = make_pipeline(TfidfVectorizer(), LogisticRegression(random_state=42, max_iter=1000))
    
    # Train the model
    model.fit(processed_patterns, tags)
    
    # Calculate training accuracy
    accuracy = model.score(processed_patterns, tags)
    print(f"Model trained successfully! Training Accuracy: {accuracy * 100:.2f}%")

    # Save the model
    print("Saving model to disk...")
    with open('chatbot_model.pkl', 'wb') as f:
        pickle.dump(model, f)
        
    print("Training complete.")

if __name__ == "__main__":
    train_model()
