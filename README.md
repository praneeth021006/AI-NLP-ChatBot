# AI Chatbot using NLP

## Project Overview
This project is an intelligent chatbot developed using Natural Language Processing (NLP) techniques and Machine Learning. It processes user text, predicts the underlying intent, and dynamically generates an appropriate response.

## Features
- **NLP Preprocessing:** Uses the NLTK library to tokenize, remove punctuation, and lemmatize conversational text data.
- **Intent Recognition:** Employs a TF-IDF Vectorizer and Logistic Regression from `scikit-learn` to classify user inputs into learned intents.
- **Modern User Interface:** Uses Streamlit to provide a beautiful, modern chat-bubble UI.
- **Customizable Dataset:** Intents and responses are easily configurable via an `intents.json` file.

## Expected Deliverables Included
1. **Chatbot source code:** `chatbot.py` and `app.py`.
2. **Training dataset:** `intents.json`.
3. **NLP preprocessing scripts:** `nlp_utils.py`.
4. **Project documentation:** This `README.md` file.

## How to Run

1. **Install dependencies:**
```bash
pip install nltk scikit-learn streamlit
```

2. **Train the NLP model:**
Run the training script to process the data and generate `chatbot_model.pkl`.
```bash
python train.py
```

3. **Launch the User Interface:**
```bash
streamlit run app.py
```
This will open the chatbot in your web browser where you can interact with it.
