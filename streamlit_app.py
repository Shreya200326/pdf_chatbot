import os
import streamlit as st
from PyPDF2 import PdfReader
import google.generativeai as genai

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    GEMINI_API_KEY = st.text_input("🔐 Enter your Gemini API key:", type="password")
    if not GEMINI_API_KEY:
        st.warning("⚠️ API key required to proceed.")
        st.stop()

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Load Gemini model (fast + free quota)
model = genai.GenerativeModel("models/gemini-1.5-flash")

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content + "\n"
    return text

# Ask Gemini with context
def ask_gemini(question, context):
    prompt = f"""You are an assistant. Answer the question using the context below.

Context:
{context}

Question:
{question}

Answer:"""
    response = model.generate_content(prompt)
    return response.text.strip()

# ---- Streamlit UI ----
st.title("📄 Gemini-Powered PDF Q&A Bot")

pdf_file = st.file_uploader("Upload your PDF", type=["pdf"])

if pdf_file:
    context = extract_text_from_pdf(pdf_file)
    st.success("✅ PDF parsed successfully!")

    question = st.text_input("Ask a question based on the document:")

    if question:
        with st.spinner("Thinking..."):
            answer = ask_gemini(question, context)
        st.markdown("**🤖 Answer:**")
        st.success(answer)
