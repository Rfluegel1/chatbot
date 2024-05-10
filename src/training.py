import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()


def load_intents_from_json(file_path):
    """
    Load and return the intents from a JSON file.
    """
    with open(file_path, 'r') as file:
        return json.load(file)['intents']


def preprocess_intents(intents_data):
    """
    Process the intents data to return tokenized words, unique tags, and tag-word pairs.
    """
    tokenized_words = []
    unique_tags = []
    tag_word_pairs = []

    for intent in intents_data:
        for pattern in intent['patterns']:
            tag = intent['tag']
            tokens = nltk.word_tokenize(pattern)
            tokenized_words.extend(tokens)
            tag_word_pairs.append((tokens, tag))
            if tag not in unique_tags:
                unique_tags.append(tag)

    return tokenized_words, unique_tags, tag_word_pairs


def process_words(words, ignore_characters=['?', '.', '!', "'"]):
    """
    Process words by lemmatizing and removing ignored characters.
    """
    processed_words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_characters]
    processed_words = sorted(set(processed_words))
    return processed_words


def create_training_data(vocabulary, tag_list, sentence_tag_pairs):
    """
    Create training data from the vocabulary, list of unique tags, and sentence-tag pairs.

    Parameters:
    - vocabulary: List of all unique words after preprocessing (lemmatization, etc.).
    - tag_list: List of all unique tags that categorize the sentences.
    - sentence_tag_pairs: List of pairs, where each pair consists of a list of words from a sentence and a corresponding tag.

    Returns:
    - input_features: Numpy array of training inputs where each entry is a bag-of-words representation of a sentence.
    - output_labels: Numpy array of output labels where each label is a one-hot encoded vector representing the tag of the sentence.
    """

    input_features = []  # Array to hold the bag-of-words representation of each sentence
    output_labels = []  # Array to hold the one-hot encoded tags for each sentence

    # Initialize a template for the one-hot encoded tag vectors
    one_hot_template = [0] * len(tag_list)

    for sentence, tag in sentence_tag_pairs:
        # Create the bag-of-words representation for the current sentence
        sentence_vector = [0] * len(vocabulary)
        lemmatized_words = [lemmatizer.lemmatize(word.lower()) for word in sentence]

        for i, vocab_word in enumerate(vocabulary):
            if vocab_word in lemmatized_words:
                sentence_vector[i] = 1

        # Create the one-hot encoded vector for the tag
        tag_vector = list(one_hot_template)
        tag_vector[tag_list.index(tag)] = 1

        # Append the vectors to their respective arrays
        input_features.append(sentence_vector)
        output_labels.append(tag_vector)

    # Convert lists to numpy arrays with type float32 for compatibility with machine learning frameworks
    input_features = np.array(input_features, dtype=np.float32)
    output_labels = np.array(output_labels, dtype=np.float32)

    return input_features, output_labels


def build_and_train_model(input_features, output_features, n_epochs=200, batch_size=5):
    """
    Build and train the neural network model.
    """
    # Create the model
    model = Sequential()
    model.add(Dense(64, input_shape=(len(input_features[0]),), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(output_features[0]), activation='softmax'))

    sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # Train the model
    model.fit(input_features, output_features, epochs=n_epochs, batch_size=batch_size, verbose=1)
    return model


def save_model_and_data(model, processed_words, unique_tags, words_filename, classes_filename, model_filename):
    """
    Save the model, processed words, and unique tags to files.

    Parameters:
    - model: Trained model to be saved.
    - processed_words: List of processed words to be saved.
    - unique_tags: List of unique intent tags to be saved.
    - words_filename: Filename for saving processed words.
    - classes_filename: Filename for saving unique tags.
    - model_filename: Filename for saving the trained model.
    """
    # Save processed words
    with open(words_filename, 'wb') as file:
        pickle.dump(processed_words, file)

    # Save unique tags
    with open(classes_filename, 'wb') as file:
        pickle.dump(unique_tags, file)

    # Save the model
    model.save(model_filename)


def main():
    # Load intents from the JSON file
    intents = load_intents_from_json('intents.json')

    # Preprocess intents to extract words, tags, and tag-word pairs
    tokenized_words, unique_tags, tag_word_pairs = preprocess_intents(intents)

    # Process words to lemmatize and remove ignored characters
    processed_words = process_words(tokenized_words)

    # Create training data from the processed words, tags, and pairs
    train_x, train_y = create_training_data(processed_words, unique_tags, tag_word_pairs)

    # Build and train the model
    model = build_and_train_model(train_x, train_y)

    # Save the model and the necessary data
    save_model_and_data(model, processed_words, unique_tags, 'words.pkl', 'classes.pkl', 'chatbot_model.keras')

    print('Model training complete!')


if __name__ == '__main__':
    main()
