# Import necessary libraries
import nltk
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from model import NeuralNet     # Custom neural network model
from nltk_utils import tokenize, bag_of_words  # NLP utility functions
import json
import torch.nn as nn

# Download NLTK tokenizer model if not already downloaded
nltk.download('punkt')

# Function to train the chatbot model
def train_model():
    # Load intents
    with open("intents.json", "r") as f:
        intents = json.load(f)

    all_patterns = []
    all_tags = []

    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            words = tokenize(pattern)
            all_patterns.append(words)
            all_tags.append(intent["tag"])

    all_words = sorted(set(w for pattern in all_patterns for w in pattern))

    X_train = [bag_of_words(pattern, all_words) for pattern in all_patterns]

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(all_tags)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    input_size = len(X_train[0])
    hidden_size = 8
    output_size = len(label_encoder.classes_)

    model = NeuralNet(input_size, hidden_size, output_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).long()

    num_epochs = 1000
    for epoch in range(num_epochs):
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Save model
    torch.save({
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "all_words": all_words,
        "tags": label_encoder.classes_,
        "model_state": model.state_dict(),
    }, "data.pth")

    print("Model training complete!")


# Entry point of the script
if __name__ == "__main__":
    train_model()
