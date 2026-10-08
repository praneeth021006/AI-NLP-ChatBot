import json
import pickle
import random
from nlp_utils import preprocess_text

class Chatbot:
    def __init__(self):
        # Load the intents dataset
        with open('intents.json', 'r', encoding='utf-8') as f:
            self.intents_data = json.load(f)
            
        # Load the trained machine learning model
        try:
            with open('chatbot_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
        except FileNotFoundError:
            raise Exception("Model not found. Please run train.py first.")

    def get_response(self, user_input):
        """
        1. Preprocess user input
        2. Predict intent
        3. Pick random response for that intent
        """
        # Preprocess
        processed_input = preprocess_text(user_input)
        
        # Predict intent
        predicted_tag = self.model.predict([processed_input])[0]
        
        # Check confidence (for Logistic Regression)
        probabilities = self.model.predict_proba([processed_input])[0]
        max_prob = max(probabilities)
        
        # If the confidence is too low, fallback
        if max_prob < 0.2:
            return "I'm sorry, I don't quite understand that."
            
        # Find the matching response
        for intent in self.intents_data['intents']:
            if intent['tag'] == predicted_tag:
                return random.choice(intent['responses'])
                
        return "I'm not sure how to answer that."

# For testing via command line
if __name__ == "__main__":
    bot = Chatbot()
    print("Chatbot initialized! Type 'quit' to exit.")
    while True:
        user_msg = input("You: ")
        if user_msg.lower() == 'quit':
            break
        response = bot.get_response(user_msg)
        print(f"Bot: {response}")
