# tag_suggester.py
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import json
import os

class TagSuggester:
    def __init__(self):
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.unknown_path = "unknown_queries.json"

    def load_unknown_queries(self):
        if os.path.exists(self.unknown_path):
            with open(self.unknown_path, "r") as f:
                data = json.load(f)
                return [q["question"] for q in data.get("queries", [])]
        else:
            return []

    def suggest_tags(self, n_clusters=5):
        queries = self.load_unknown_queries()
        if not queries:
            return []

        embeddings = self.model.encode(queries)

        # Group queries into clusters
        kmeans = KMeans(n_clusters=min(n_clusters, len(queries)), random_state=42)
        labels = kmeans.fit_predict(embeddings)

        clustered = {}
        for label, query in zip(labels, queries):
            clustered.setdefault(label, []).append(query)

        suggested_tags = []
        for cluster_id, questions in clustered.items():
            # Suggest a tag name based on first question in cluster
            sample_question = questions[0]
            words = sample_question.lower().split()[:2]  # Take first 2 words
            tag_name = "_".join(words)
            suggested_tags.append({
                "tag": tag_name,
                "examples": questions
            })

        return suggested_tags
