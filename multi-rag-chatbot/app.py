import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, WikipediaLoader, YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_ollama import OllamaEmbeddings
import tempfile

# Loaders
def load_pdf(uploaded_file):
    if uploaded_file is None:
        return []
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        loader = PyPDFLoader(tmp.name)
        return loader.load()

def load_web(url):
    return WebBaseLoader(url).load() if url else []

def load_wiki(query):
    return WikipediaLoader(query=query, load_max_docs=1).load() if query else []

def load_youtube(url):
    return YoutubeLoader.from_youtube_url(url, add_video_info=False).load() if url else []

# Core Functions
def split_docs(docs, chunk_size=500, chunk_overlap=50):
    return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap).split_documents(docs)

def get_embeddings():
    return OllamaEmbeddings(model="llama3")

def get_llm():
    return Ollama(model="llama3")

def create_qa_chain(docs):
    chunks = split_docs(docs)
    db = FAISS.from_documents(chunks, get_embeddings())
    retriever = db.as_retriever()
    return RetrievalQA.from_chain_type(llm=get_llm(), retriever=retriever)

# Streamlit App
st.set_page_config(page_title="📚 RAG Chatbot", layout="centered")
st.title("🧠 Multi-Source RAG Chatbot")

# Selection Menu
option = st.selectbox("📌 Select a source type", ["Select...", "Upload PDF", "Website URL", "Wikipedia Topic", "YouTube URL"])

docs = []

if option == "Upload PDF":
    uploaded_file = st.file_uploader("📄 Upload your PDF", type=["pdf"])
    if uploaded_file and st.button("📥 Load PDF"):
        with st.spinner("Loading PDF..."):
            docs = load_pdf(uploaded_file)
            st.session_state.qa = create_qa_chain(docs)
            st.success("✅ PDF loaded and ready!")

elif option == "Website URL":
    url = st.text_input("🌐 Enter Website URL")
    if url and st.button("📥 Load Website"):
        with st.spinner("Loading website..."):
            docs = load_web(url)
            st.session_state.qa = create_qa_chain(docs)
            st.success("✅ Website content loaded!")

elif option == "Wikipedia Topic":
    topic = st.text_input("📖 Enter Wikipedia topic")
    if topic and st.button("📥 Load Wikipedia"):
        with st.spinner("Loading Wikipedia..."):
            docs = load_wiki(topic)
            st.session_state.qa = create_qa_chain(docs)
            st.success("✅ Wikipedia content loaded!")

elif option == "YouTube URL":
    yt_url = st.text_input("📺 Enter YouTube video URL")
    if yt_url and st.button("📥 Load YouTube Transcript"):
        with st.spinner("Loading YouTube transcript..."):
            docs = load_youtube(yt_url)
            st.session_state.qa = create_qa_chain(docs)
            st.success("✅ YouTube transcript loaded!")

# Chatbot UI
if "qa" in st.session_state:
    query = st.text_input("💬 Ask your question")
    if query:
        with st.spinner("Thinking..."):
            response = st.session_state.qa.run(query)
            st.markdown(f"🤖 **Answer:** {response}")
else:
    st.info("ℹ️ Please load a source to enable the chatbot.")
