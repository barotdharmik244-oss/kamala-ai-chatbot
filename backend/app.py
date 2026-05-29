from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

app = Flask(__name__)
CORS(app)

# Load Excel File
df = pd.read_excel("../data/knowledge_base.xlsx")

# Clean Data
df.columns = df.columns.str.strip()

df = df.dropna(subset=["Question", "Answer"])

questions = (
    df["Question"]
    .fillna("")
    .astype(str)
    .str.lower()
    .str.strip()
    .tolist()
)

answers = (
    df["Answer"]
    .fillna("")
    .astype(str)
    .tolist()
)

answers = (
    df["Answer"]
    .astype(str)
    .tolist()
)

print("Knowledge Base Loaded Successfully")
print("Total Questions:", len(questions))

# Load Multilingual Model
model = SentenceTransformer(
    'paraphrase-multilingual-MiniLM-L12-v2'
)

# Create Embeddings
question_embeddings = model.encode(
    questions,
    convert_to_numpy=True
)

# Convert to float32
question_embeddings = np.array(
    question_embeddings
).astype("float32")

# Normalize
faiss.normalize_L2(question_embeddings)

# Create FAISS Index
dimension = question_embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(question_embeddings)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask_question():

    data = request.json

    user_question = (
        data.get("question", "")
        .lower()
        .strip()
    )

    print("User Question:", user_question)

    # User Embedding
    user_embedding = model.encode(
        [user_question],
        convert_to_numpy=True
    )

    user_embedding = np.array(
        user_embedding
    ).astype("float32")

    faiss.normalize_L2(user_embedding)

    # Search
    D, I = index.search(user_embedding, k=1)

    similarity_score = D[0][0]

    best_match_index = I[0][0]

    print("Similarity:", similarity_score)

    # Threshold
    if similarity_score > 0.40:

        answer = answers[best_match_index]

    else:

        answer = (
            "Sorry, I could not find the relevant information. "
            "Please contact Kamala Paints and Hardware "
            "for further assistance."
        )

    return jsonify({
        "question": user_question,
        "answer": answer
    })

import os

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port
    )