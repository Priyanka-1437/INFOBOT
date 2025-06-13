import random
import json
import pickle

model = pickle.load(open("chatbot_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

with open("chatbot_intents.json",encoding="utf-8") as f:
    chatbot_intents = json.load(f)

def get_response(user_input, threshold=0.0):
    X = vectorizer.transform([user_input.lower()])
    
    # Get prediction probabilities
    probs = model.predict_proba(X)[0]
    predicted_index = probs.argmax()
    predicted_tag = model.classes_[predicted_index]
    confidence = probs[predicted_index]
    
    if confidence < threshold:
        return "Sorry, I didn't understand that. Could you please rephrase?"
    
    for intent in chatbot_intents["intents"]:
        if intent["tag"] == predicted_tag:
            return random.choice(intent["responses"])
    
    return "Sorry, I didn't understand that."
