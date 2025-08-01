import os
import streamlit as st
from PyPDF2 import PdfReader
import google.generativeai as genai
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
import tempfile

# ------------------ SETUP ------------------

st.set_page_config(page_title="Gemini RAG PDF Chatbot")

st.title("📄 Gemini RAG PDF Chatbot (Streamlit Cloud Compatible)")

# 🔐 Load Gemini API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    GEMINI_API_KEY = st.text_input("🔐 Enter your Gemini API key:", type="password")
    if not GEMINI_API_KEY:
        st.warning("⚠️ Gemini API key is required.")
        st.stop()

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("models/gemini-1.5-flash")

# 🔐 Load Hugging Face API Key
HF_API_KEY = os.getenv("HF_API_KEY")
if not HF_API_KEY:
    HF_API_KEY = st.text_input("🧠 Enter your Hugging Face API key:", type="password")
    if not HF_API_KEY:
        st.warning("⚠️ Hugging Face API key is required.")
        st.stop()

# Embedding model (streamlit cloud compatible)
embedder = HuggingFaceInferenceAPIEmbeddings(
    api_key=HF_API_KEY,
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

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

def build_chroma_index(chunks):
    docs = [Document(page_content=chunk) for chunk in chunks]
    vectorstore = Chroma.from_documents(
        docs, 
        embedding=embedder, 
        persist_directory=tempfile.mkdtemp()
    )
    return vectorstore

def retrieve_top_k(query, vectorstore, k=4):
    results = vectorstore.similarity_search(query, k=k)
    return [doc.page_content for doc in results]

def ask_gemini(query, top_chunks):
    context = "\n\n".join(top_chunks)
    prompt = f"""Use the context below to answer the question.

Context:
{context}

Question:
{query}

Answer:"""
    response = gemini_model.generate_content(prompt)
    return response.text.strip()

# ------------------ STREAMLIT UI ------------------

pdf_file = st.file_uploader("📄 Upload your PDF", type=["pdf"])

if pdf_file:
    with st.spinner("🔄 Processing PDF..."):
        text = extract_text_from_pdf(pdf_file)
        chunks = split_into_chunks(text)
        vectorstore = build_chroma_index(chunks)
        st.success(f"✅ Document indexed with {len(chunks)} chunks.")

    query = st.text_input("💬 Ask a question based on the PDF:")

    if query:
        with st.spinner("🤖 Thinking..."):
            top_chunks = retrieve_top_k(query, vectorstore)
            answer = ask_gemini(query, top_chunks)
        st.markdown("### 🤖 Answer")
        st.success(answer)
