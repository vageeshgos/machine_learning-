import streamlit as st

st.set_page_config(page_title="AI Q&A Chatbot", layout="centered")

st.title("🤖 AI Q&A Chatbot")
st.write("Ask me anything about Artificial Intelligence (A-Z)!")

# Input box
query = st.text_input("🔍 Your Question")

# Dummy answer logic (replace with your embedding-based answer logic)
if query:
    # You can replace this with document similarity logic
    st.success(f"📘 Answer: I’ll soon answer your question: **{query}**")

from flask import Flask, render_template, request
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# Load documents and embed them once
documents = [
    "Artificial Intelligence (AI) is the simulation of human intelligence in machines.",
    "Machine learning is a subset of AI that uses statistical techniques to give machines the ability to learn from data.",
    "Deep learning uses neural networks with many layers to model complex patterns.",
    "Natural Language Processing enables computers to understand human language.",
    "Computer Vision allows machines to analyze and interpret images and videos.",
    "ChatGPT is an AI chatbot based on the GPT model developed by OpenAI.",
    "Transformers are a type of neural network architecture used in NLP.",
    "Gemini is a multimodal model developed by Google combining vision and language.",
    "Claude is an AI assistant developed by Anthropic.",
    "AI agents can make decisions and take actions to complete a task."
]

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
doc_embeddings = embedding.embed_documents(documents)

@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    if request.method == "POST":
        query = request.form["query"]
        query_embedding = embedding.embed_query(query)
        similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
        top_indices = np.argsort(similarities)[-3:][::-1]
        results = [(documents[i], float(similarities[i])) for i in top_indices]
    return render_template("index.html", results=results)

if __name__ == "__main__":
    app.run(debug=True)
