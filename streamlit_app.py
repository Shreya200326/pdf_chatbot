import os
import streamlit as st
from PyPDF2 import PdfReader
import google.generativeai as genai
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import CharacterTextSplitter

# ------------------ SETUP ------------------

# Load Gemini API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    GEMINI_API_KEY = st.text_input("🔐 Enter your Gemini API key:", type="password")
    if not GEMINI_API_KEY:
        st.warning("API key required to proceed.")
        st.stop()

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("models/gemini-1.5-flash")

# Embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------ FUNCTIONS ------------------

def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content + "\n"
    return text

def split_into_chunks(text, chunk_size=1000, overlap=200):
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)

def build_faiss_index(chunks):
    embeddings = embed_model.encode(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index, embeddings

def retrieve_top_k(query, chunks, index, k=4):
    query_embedding = embed_model.encode([query])
    _, indices = index.search(np.array(query_embedding), k)
    return [chunks[i] for i in indices[0]]

def ask_gemini(query, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"""Use the context below to answer the question.

Context:
{context}

Question:
{query}

Answer:"""
    response = gemini_model.generate_content(prompt)
    return response.text.strip()

# ------------------ STREAMLIT UI ------------------

st.title("📄 Gemini RAG PDF Chatbot (Level 1)")

pdf = st.file_uploader("Upload your PDF", type=["pdf"])

if pdf:
    with st.spinner("Extracting and processing..."):
        text = extract_text_from_pdf(pdf)
        chunks = split_into_chunks(text)
        faiss_index, _ = build_faiss_index(chunks)
        st.success(f"✅ Document indexed with {len(chunks)} chunks.")

    user_query = st.text_input("Ask a question based on the document:")

    if user_query:
        with st.spinner("Searching and answering..."):
            top_chunks = retrieve_top_k(user_query, chunks, faiss_index)
            answer = ask_gemini(user_query, top_chunks)
        st.markdown("### 🤖 Answer")
        st.success(answer)
