
# 📄 Gemini RAG PDF Chatbot (Streamlit)

A simple yet powerful Retrieval-Augmented Generation (RAG) chatbot built using Google Gemini and Hugging Face embeddings to answer questions based on the content of uploaded PDF documents.

---

# 🚀 Features

✅ Upload and parse PDFs  
✅ Chunk and embed text using Hugging Face sentence transformers  
✅ Perform semantic search using Chroma vector store  
✅ Ask natural language questions and get AI-generated answers using Google Gemini  
---

# 🧠 How It Works

1. **PDF Upload**: Upload one PDF file via the UI.  
2. **Text Extraction**: Extract text from PDF pages.  
3. **Chunking**: Split extracted text into overlapping chunks.  
4. **Embedding**: Convert text chunks into vector embeddings using Hugging Face API.  
5. **Vector Store**: Store and search embeddings using Chroma.  
6. **Query Answering**: Retrieve top-k similar chunks and pass them to Gemini to generate a context-aware answer.

---
# 🔧 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/gemini-rag-pdf-chatbot.git
cd gemini-rag-pdf-chatbot
````

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> 📝 If using Streamlit Cloud, just include this `requirements.txt` in your repo. It will install everything automatically.

### 3. Environment Variables

Create a `.env` file or set the following environment variables:

```
GEMINI_API_KEY=your_google_gemini_api_key
HF_API_KEY=your_huggingface_api_key
```

> If you don’t want to use `.env`, the app will prompt you to enter keys manually through the UI.

### 4. Run the App

```bash
streamlit run app.py
```

---

## 📁 File Structure

```
├── app.py                # Main Streamlit app
├── requirements.txt      # Python dependencies
└── README.md             # You're reading it
```

---

## 🧪 Example Models Used

* Google Gemini: `models/gemini-1.5-flash`
* Hugging Face: `sentence-transformers/all-MiniLM-L6-v2`

---

## 📦 Dependencies

Key packages used:

* `streamlit`
* `PyPDF2`
* `langchain`
* `google-generativeai`
* `HuggingFaceInferenceAPIEmbeddings`
* `Chroma` (Vector store)
* `python-dotenv` (optional)

> All specified in `requirements.txt`.


---

## 💡 Future Improvements

* [ ] Support for multiple PDFs
* [ ] Add conversation memory
* [ ] Metadata-based filtering (doc name, page number)
* [ ] Agent-based orchestration
* [ ] UI enhancements (dark mode, document preview, etc.)


```
