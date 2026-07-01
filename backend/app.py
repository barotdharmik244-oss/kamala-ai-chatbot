from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

import google.generativeai as genai

genai.configure(
    api_key=os.environ.get("AQ.Ab8RN6LOYG88E_mIF-NGkxiJ2VoawnzbRcmSu6aMMm1p1DN2GA")
)
genai.configure(api_key=os.environ.get("AQ.Ab8RN6LOYG88E_mIF-NGkxiJ2VoawnzbRcmSu6aMMm1p1DN2GA"))  # ⚠️ Move to env variable in production
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

app = Flask(__name__)
CORS(app)

# Load Excel File
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

excel_path = os.path.join(BASE_DIR, "..", "data", "knowledge_base.xlsx")

df = pd.read_excel(excel_path)
print(df.columns.tolist())

# Clean Data
df.columns = df.columns.str.strip()

df = df.dropna(subset=["Question (English)", "Answer (English)"])

questions_en = (
    df["Question (English)"]
    .fillna("")
    .astype(str)
    .str.lower()
    .str.strip()
)

questions_gu = (
    df["Question (Gujarati)"]
    .fillna("")
    .astype(str)
    .str.lower()
    .str.strip()
)

questions = (questions_en + " " + questions_gu).tolist()


print("Knowledge Base Loaded Successfully")
print("Total Questions:", len(questions))

# Load Multilingual Model
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Create Embeddings
question_embeddings = model.encode(questions, convert_to_numpy=True)

# Convert to float32 and Normalize
question_embeddings = np.array(question_embeddings).astype("float32")
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
    user_question = data.get("question", "").lower().strip()

    print("User Question:", user_question)

    # User Embedding
    user_embedding = model.encode([user_question], convert_to_numpy=True)
    user_embedding = np.array(user_embedding).astype("float32")
    
    faiss.normalize_L2(user_embedding)

    # Search
    D, I = index.search(user_embedding, k=1)

    similarity_score = D[0][0]
    best_match_index = I[0][0]

    print("Similarity:", similarity_score)

    # Threshold check — FIX: return is now inside both branches
    if similarity_score > 0.40:
        english_answer = df.iloc[best_match_index]["Answer (English)"]
        gujarati_answer = df.iloc[best_match_index]["Answer (Gujarati)"]

        # Detect Gujarati script (Unicode range \u0A80–\u0AFF)
        if any('\u0A80' <= c <= '\u0AFF' for c in user_question):
            answer = gujarati_answer
        else:
            answer = english_answer

        prompt = f"""
You are an AI assistant for Kamala Paints and Hardware.

Customer Question:
{user_question}

Knowledge Base Information:
{answer}

Instructions:
- Answer naturally and professionally.
- Use ONLY the knowledge base information provided above.
- Do not invent new information.
- Reply in the same language used by the customer.
- Ask one short follow-up question related to paints or hardware.
"""
        gemini_response = gemini_model.generate_content(prompt)
        answer = gemini_response.text

    else:
        answer = (
            "Sorry, I could not find the relevant information. "
            "Please contact Kamala Paints and Hardware for further assistance."
        )

    # FIX: return is now at the correct indentation level (inside the function, after if/else)
    return jsonify({
        "question": user_question,
        "answer": answer
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)