# Import necessary libraries
from flask import Flask, render_template, request, jsonify, redirect
import random
import json
import torch
import subprocess
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import datetime
import os
from tag_suggester import TagSuggester

suggester = TagSuggester()

# Initialize Flask app
app = Flask(__name__)
# Set device to GPU if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Function to load intents and the trained model
def load_intents_and_model():
    # Load intents data
    with open('intents.json', 'r') as json_data:
        intents = json.load(json_data)

    # Load model data
    FILE = "data.pth"
    data = torch.load(FILE, weights_only=False)  # <-- Fix is here!

    # Extract necessary data
    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data["all_words"]
    tags = data["tags"]
    model_state = data["model_state"]

    # Initialize and load the model
    model = NeuralNet(input_size, hidden_size, output_size)
    model.load_state_dict(model_state)
    model.eval()

    return intents, model, all_words, tags


# Load model and intents when the app starts
global intents, model, all_words, tags
intents, model, all_words, tags = load_intents_and_model()

# Function to generate a chatbot response based on user message
def get_response(msg):
    # Tokenize and vectorize the message
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(device)

    # Get model prediction
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    # Calculate probability
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    # If the prediction confidence is high enough
    if prob.item() > 0.7:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                response = random.choice(intent['responses']) # Randomly picks one response
                log_interaction(msg, response)  # Log successful interaction
                return response
    else:
        print(f"[DEBUG] Unknown query: {msg}")
        log_unknown_query(msg)  # Log unknown query
        # Fallback to a default response
        response = get_fallback_response(msg)
        log_interaction(msg, response)
        return response

# Function to fetch a fallback response
def get_fallback_response(msg):
    for intent in intents['intents']:
        if intent["tag"] == "fallback":
            return random.choice(intent["responses"])
    return "I'm not sure I understand. Can you please rephrase?"

def log_interaction(user_input, bot_response):
    timestamp = datetime.datetime.now().isoformat()

    # Logs
    logs_path = "logs.json"
    if os.path.exists(logs_path):
        with open(logs_path, "r") as f:
            logs_data = json.load(f)
    else:
        logs_data = {"conversations": []}
    
    logs_data["conversations"].append({
        "timestamp": timestamp,
        "user": user_input,
        "bot": bot_response
    })
    
    with open(logs_path, "w") as f:
        json.dump(logs_data, f, indent=4)


# Function to log unknown queries for future improvements
def log_unknown_query(user_input):
    unknown_path = "unknown_queries.json"

    # Try to load existing unknown queries
    if os.path.exists(unknown_path):
        try:
            with open(unknown_path, "r") as f:
                unknown_data = json.load(f)
        except json.JSONDecodeError:
            unknown_data = {"queries": []}
    else:
        unknown_data = {"queries": []}

    # Append the new unknown query
    unknown_data["queries"].append({
        "timestamp": datetime.datetime.now().isoformat(),
        "question": user_input
    })

    # Save updated unknown queries
    with open(unknown_path, "w") as f:
        json.dump(unknown_data, f, indent=4)

    print(f"[DEBUG] Saved unknown query to {unknown_path}")

def remove_known_questions_from_unknown():
    # Load intents
    with open("intents.json", "r") as f:
        intents_data = json.load(f)

    # Gather all patterns from intents
    known_patterns = set()
    for intent in intents_data["intents"]:
        for pattern in intent["patterns"]:
            known_patterns.add(pattern.strip().lower())

    # Load unknown queries
    unknown_path = "unknown_queries.json"
    if os.path.exists(unknown_path):
        with open(unknown_path, "r") as f:
            unknown_data = json.load(f)
    else:
        print("[DEBUG] No unknown_queries.json file found.")
        return

    # Filter unknown queries
    filtered_queries = []
    for query in unknown_data.get("queries", []):
        question_text = query["question"].strip().lower()
        if question_text not in known_patterns:
            filtered_queries.append(query)

    # Save back the cleaned unknown queries
    with open(unknown_path, "w") as f:
        json.dump({"queries": filtered_queries}, f, indent=4)

    print(f"[DEBUG] Removed {len(unknown_data.get('queries', [])) - len(filtered_queries)} known queries from unknown_queries.json.")

@app.route("/clean_unknowns")
def clean_unknowns():
    remove_known_questions_from_unknown()
    return "Unknown queries cleaned successfully!"

# Admin page
@app.route("/admin")
def admin_page():
    suggested_tags = suggester.suggest_tags(n_clusters=5)
    return render_template("admin.html", suggested_tags=suggested_tags)

@app.route("/approve_tag", methods=["POST"])
def approve_tag():
    tag = request.form["tag"]
    examples = request.form["examples"].split("||")
    response = request.form["response"]

    with open("intents.json", "r") as f:
        data = json.load(f)

    new_intent = {
        "tag": tag,
        "patterns": examples,
        "responses": [response]  # <-- save admin's entered response
    }

    data["intents"].append(new_intent)

    with open("intents.json", "w") as f:
        json.dump(data, f, indent=4)

    # Optionally retrain model
    subprocess.call(["python", "train.py"])

    # Reload intents and model
    global intents, model, all_words, tags
    intents, model, all_words, tags = load_intents_and_model()

    return redirect("/admin")

# Flask route: Home page
@app.route("/")
def index():
    return render_template("index.html")

# Flask route: API endpoint for getting chatbot response
@app.route("/get", methods=["POST"])
def chatbot_response():
    data = request.get_json()
    msg = data.get("message")
    response = get_response(msg)
    return jsonify({"reply": response})

# Flask route: Page to train the model (form to submit new intents)
@app.route("/train")
def train_page():
    return render_template("train.html")

# Flask route: Handle new intent submission and retrain the model
@app.route("/submit_intent", methods=["POST"])
def submit_intent():
    tag = request.form["tag"]
    patterns = request.form["patterns"].split("\n")
    responses = request.form["responses"].split("\n")

    # Load existing intents
    with open("intents.json", "r") as f:
        data = json.load(f)

    # Add the new intent
    data["intents"].append({
        "tag": tag,
        "patterns": [p.strip() for p in patterns if p.strip()],
        "responses": [r.strip() for r in responses if r.strip()]
    })

    # Save updated intents
    with open("intents.json", "w") as f:
        json.dump(data, f, indent=4)

    # Retrain model using the updated intents
    subprocess.call(["python", "train.py"])

    # Reload updated model and intents into memory
    global intents, model, all_words, tags
    intents, model, all_words, tags = load_intents_and_model()

    return redirect("/")

# Run Flask app
if __name__ == "__main__":
    remove_known_questions_from_unknown()
    app.run(debug=True)
