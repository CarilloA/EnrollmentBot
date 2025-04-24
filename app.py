from flask import Flask, render_template, request, jsonify, redirect
import random
import json
import torch
import subprocess
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

app = Flask(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents and model
def load_intents_and_model():
    with open('intents.json', 'r') as json_data:
        intents = json.load(json_data)

    FILE = "data.pth"
    data = torch.load(FILE)

    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data["all_words"]
    tags = data["tags"]
    model_state = data["model_state"]

    model = NeuralNet(input_size, hidden_size, output_size)
    model.load_state_dict(model_state)
    model.eval()

    return intents, model, all_words, tags

# Load model on app start
intents, model, all_words, tags = load_intents_and_model()

# Function to get a response
def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.7:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])

    return "I'm sorry, I didn't understand that. Please try again."

# Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    data = request.get_json()
    msg = data.get("message")
    response = get_response(msg)
    return jsonify({"reply": response})

@app.route("/train")
def train_page():
    return render_template("train.html")

@app.route("/submit_intent", methods=["POST"])
def submit_intent():
    tag = request.form["tag"]
    patterns = request.form["patterns"].split("\n")
    responses = request.form["responses"].split("\n")

    # Load existing intents
    with open("intents.json", "r") as f:
        data = json.load(f)

    # Add new intent
    data["intents"].append({
        "tag": tag,
        "patterns": [p.strip() for p in patterns if p.strip()],
        "responses": [r.strip() for r in responses if r.strip()]
    })

    # Save updated intents
    with open("intents.json", "w") as f:
        json.dump(data, f, indent=4)

    # Retrain model
    subprocess.call(["python", "train.py"])

    # Reload updated model
    global intents, model, all_words, tags
    intents, model, all_words, tags = load_intents_and_model()

    return redirect("/")

if __name__ == "__main__":
    app.run(debug=True)
