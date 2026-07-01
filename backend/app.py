from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

import pandas as pd
import os

import google.generativeai as genai

genai.configure(
    api_key=os.environ.get("GEMINI_API_KEY")
)

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


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask_question():

    data = request.get_json(silent=True) or {}
    user_question = str(data.get("question", "")).lower().strip()

    print("User Question:", user_question)

    if not user_question:
        return jsonify({
            "question": user_question,
            "answer": "Please type a question."
        })

    best_match_index = -1

    for idx, row in df.iterrows():

        english_q = str(row["Question (English)"]).lower()
        gujarati_q = str(row["Question (Gujarati)"]).lower()

        if (
            user_question in english_q
            or english_q in user_question
            or user_question in gujarati_q
            or gujarati_q in user_question
        ):
            best_match_index = idx
            break

    if best_match_index != -1:
        english_answer = df.iloc[best_match_index]["Answer (English)"]
        gujarati_answer = df.iloc[best_match_index]["Answer (Gujarati)"]

        # Detect Gujarati script (Unicode range \u0A80–\u0AFF)
        if any('\u0A80' <= c <= '\u0AFF' for c in user_question):
            kb_answer = gujarati_answer
        else:
            kb_answer = english_answer

        prompt = f"""
You are an AI assistant for Kamala Paints and Hardware.

Customer Question:
{user_question}

Knowledge Base Information:
{kb_answer}

Instructions:
- Answer naturally and professionally.
- Use ONLY the knowledge base information provided above.
- Do not invent new information.
- Reply in the same language used by the customer.
- Ask one short follow-up question related to paints or hardware.
"""
        try:
            gemini_response = gemini_model.generate_content(prompt)
            answer = gemini_response.text
        except Exception as e:
            print("Gemini API Error:", e)
            answer = kb_answer  # fallback to raw knowledge base answer

    else:
        answer = (
            "Sorry, I could not find the relevant information. "
            "Please contact Kamala Paints and Hardware for further assistance."
        )

    return jsonify({
        "question": user_question,
        "answer": answer
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(
        host="0.0.0.0",
        port=port
    )