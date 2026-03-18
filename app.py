import streamlit as st
from dotenv import load_dotenv
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

# ---------------- LOAD API ----------------
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")

client = MistralClient(api_key=api_key)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Notes Assistant", layout="wide")

# ---------------- MODERN UI ----------------
st.markdown("""
<style>

/* Background */
body {
    background-color: #0f172a;
}

/* Title */
h1 {
    color: #38bdf8;
}

/* Input */
.stTextInput input {
    background-color: #1e293b !important;
    color: white !important;
    border-radius: 8px;
}

/* Button */
.stButton button {
    background-color: #38bdf8;
    color: black;
    font-weight: bold;
    border-radius: 10px;
    padding: 10px;
}

/* Answer box */
.answer-box {
    padding: 20px;
    border-radius: 12px;
    margin-top: 15px;
    color: white;
    font-size: 16px;
    line-height: 1.6;
}

/* PDF Answer */
.pdf-box {
    background-color: #1e293b;
    border-left: 6px solid #22c55e;
}

/* AI Answer */
.bot-box {
    background-color: #1e293b;
    border-left: 6px solid #f59e0b;
}

/* Headings */
.answer-box b {
    color: #e2e8f0;
    font-size: 18px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("AI Notes Assistant")

# ---------------- SESSION ----------------
if "db" not in st.session_state:
    st.session_state.db = None
if "processed" not in st.session_state:
    st.session_state.processed = False

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file and not st.session_state.processed:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF uploaded successfully!")

    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings()
    st.session_state.db = FAISS.from_documents(chunks, embeddings)

    st.session_state.processed = True
    st.success("Ready! Ask your question below ")

# ---------------- QUESTION UI ----------------
if st.session_state.processed:
    col1, col2 = st.columns([4,1])

    with col1:
        query = st.text_input(" Ask anything from your notes:")

    with col2:
        ask_btn = st.button("Ask ")

    if ask_btn and query:

        # -------- PDF ANSWER --------
        docs = st.session_state.db.similarity_search(query, k=3)
        context = "\n".join([doc.page_content for doc in docs])

        prompt_pdf = f"""
        Answer ONLY using this context.
        If not found, say "Not found in document".

        Context:
        {context}

        Question:
        {query}
        """

        response_pdf = client.chat(
            model="mistral-small",
            messages=[ChatMessage(role="user", content=prompt_pdf)]
        )

        pdf_answer = response_pdf.choices[0].message.content

        # -------- GENERAL ANSWER --------
        prompt_general = f"""
        Answer the question using your general knowledge:

        {query}
        """

        response_general = client.chat(
            model="mistral-small",
            messages=[ChatMessage(role="user", content=prompt_general)]
        )

        general_answer = response_general.choices[0].message.content

        # -------- DISPLAY --------
        st.markdown("## Answers")

        st.markdown(f"""
        <div class="answer-box pdf-box">
        <b>Answer from your PDF:</b><br><br>
        {pdf_answer.replace('\n','<br>')}
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="answer-box bot-box">
        <b>General AI Answer:</b><br><br>
        {general_answer.replace('\n','<br>')}
        </div>
        """, unsafe_allow_html=True)