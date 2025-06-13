import json
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import os

nltk.download('punkt')

with open("chatbot_intents.json","r",encoding ="utf-8") as f:
    data = json.load(f)

corpus = []
labels = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        corpus.append(pattern.lower())
        labels.append(intent["tag"])

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

model = MultinomialNB()
model.fit(X, labels)

# Save model and vectorizer
with open("chatbot_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
