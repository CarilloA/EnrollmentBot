# Import necessary libraries
import nltk
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from model import NeuralNet     # Custom neural network model
from nltk_utils import tokenize, bag_of_words  # NLP utility functions
import json

# Download NLTK tokenizer model if not already downloaded
nltk.download('punkt')

# Function to train the chatbot model
def train_model():
    # Load intents file (contains patterns and responses)
    with open("intents.json", "r") as f:
        intents = json.load(f)

    all_patterns = []   # List to store tokenized patterns
    all_tags = []   # List to store corresponding tags

    # Iterate over each intent in the intents file
    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            words = tokenize(pattern)  # Tokenize each pattern into words
            all_patterns.append(words)  # Store tokenized pattern
            all_tags.append(intent["tag"])  # Store corresponding tag

    # Prepare the vocabulary (all unique words) for training
    all_words = sorted(set(w for pattern in all_patterns for w in pattern))

    # Create input feature vectors (bag of words) for each pattern
    X_train = [bag_of_words(pattern, all_words) for pattern in all_patterns]

    # Encode the tags into numerical labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(all_tags)

    # Convert training data to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Define the size of input, hidden, and output layers
    input_size = len(X_train[0])   # No. of input features (words)
    hidden_size = 8    # No. of neurons in hidden layer
    output_size = len(label_encoder.classes_)  # No. of output classes (tags)

    # Initialize the neural network model
    model = NeuralNet(input_size, hidden_size, output_size)

    # Train the model using the training data
    model.train_model(X_train, y_train)

    # Save the trained model and related metadata to a file
    torch.save({
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "all_words": all_words,
        "tags": label_encoder.classes_,
        "model_state": model.state_dict(),  # Save model parameters
    }, "data.pth")

    print("Model training complete!")  # Print completion message

# Entry point of the script
if __name__ == "__main__":
    train_model()
