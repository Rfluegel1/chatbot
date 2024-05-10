import random
import json
import pickle
import numpy
import nltk

from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()


def load_intents_from_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


intents = load_intents_from_json('intents.json')

processed_words = pickle.load(open('words.pkl', 'rb'))
unique_tags = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.keras')


def preprocess_sentence(sentence):
    # Tokenize and lemmatize the given sentence
    tokens = nltk.word_tokenize(sentence)
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens


def create_bag_of_words(sentence):
    # Convert the sentence into a bag of words for model prediction
    tokens = preprocess_sentence(sentence)
    bag = [0] * len(processed_words)
    for token in tokens:
        for i, word in enumerate(processed_words):
            if word == token:
                bag[i] = 1
    return numpy.array(bag)


def predict_intent(sentence):
    # Predict the class of the sentence using the trained model
    bag = create_bag_of_words(sentence)
    predictions = model.predict(numpy.array([bag]))[0]
    error_threshold = 0.25
    results = [[i, r] for i, r in enumerate(predictions) if r > error_threshold]
    results.sort(key=lambda x: x[1], reverse=True)

    predicted_intents = []
    for result in results:
        predicted_intents.append({'intent': unique_tags[result[0]], 'probability': str(result[1])})
    return predicted_intents


def get_response(predicted_intents, intents_json):
    # Fetch a response from the matched intent
    tag = predicted_intents[0]['intent']
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])


# Main chatbot loop
while True:
    message = input('')
    if message == "quit":
        print("Chatbot exiting.")
        break

    predicted_intents = predict_intent(message)
    response = get_response(predicted_intents, intents)
    print(response)

