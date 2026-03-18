# DocuMind AI

An AI-powered document assistant that allows users to upload PDF files and ask questions using natural language. The system uses Retrieval-Augmented Generation (RAG) to provide accurate answers from the document along with general AI responses.

---

## 🚀 Features

- 📄 Upload and process PDF documents
- 💬 Ask questions in natural language
- 🧠 Dual-answer system:
  - Context-based answer (from PDF)
  - General AI answer (model knowledge)
- ⚡ Fast semantic search using FAISS
- 🎨 Clean and modern UI using Streamlit

---

## 🧠 How It Works

1. Upload a PDF document
2. The document is split into smaller chunks
3. Chunks are converted into embeddings
4. Stored in FAISS vector database
5. User asks a question
6. Relevant chunks are retrieved
7. AI generates:
   - Answer from PDF (RAG)
   - General answer (LLM)

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit  
- **Backend:** Python  
- **LLM:** Mistral API  
- **Embeddings:** HuggingFace  
- **Vector Database:** FAISS  
- **Framework:** LangChain  

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Subhadra-Bhattacharyya/documind-ai.git
cd documind-ai
