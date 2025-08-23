import os
import streamlit as st
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# -----------------------
# Load API Key
# -----------------------
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("⚠️ OPENAI_API_KEY not found in .env file")
    st.stop()

os.environ["OPENAI_API_KEY"] = openai_api_key

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="News Research Assistant", layout="wide")
st.title("📰 News Research Assistant")

# -----------------------
# Embeddings + FAISS
# -----------------------
embeddings = OpenAIEmbeddings()
VECTORSTORE_DIR = "faiss_store"

if os.path.exists(VECTORSTORE_DIR):
    vectorstore = FAISS.load_local(VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True)
else:
    vectorstore = None

# -----------------------
# Helper: fetch text from URL
# -----------------------
def extract_text_from_url(url):
    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")

        # remove scripts/styles
        for script in soup(["script", "style"]):
            script.extract()

        text = " ".join(soup.stripped_strings)
        return text
    except Exception as e:
        return f"Error fetching {url}: {str(e)}"

# -----------------------
# Upload URLs
# -----------------------
st.subheader("🔗 Add News Articles")
urls_input = st.text_area("Enter one or more URLs (comma or newline separated):")

if st.button("Process URLs"):
    if urls_input.strip():
        urls = [u.strip() for u in urls_input.replace(",", "\n").split("\n") if u.strip()]
        texts = [extract_text_from_url(u) for u in urls]

        if vectorstore is None:
            vectorstore = FAISS.from_texts(texts, embedding=embeddings)
        else:
            vectorstore.add_texts(texts)

        vectorstore.save_local(VECTORSTORE_DIR)
        st.success(f"✅ Added {len(texts)} documents to the index.")

# -----------------------
# Retriever + LLM
# -----------------------
if vectorstore:
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.3,
        max_tokens=500,
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )

    # -----------------------
    # User Q&A
    # -----------------------
    query = st.text_input("Ask me anything about the indexed news/articles:")

    if query:
        with st.spinner("Thinking..."):
            result = qa_chain({"question": query})

        st.markdown("### 📌 Answer")
        st.write(result["answer"])

        st.markdown("### 📚 Sources")
        for i, doc in enumerate(result["source_documents"], 1):
            st.write(f"**Source {i}:** {doc.page_content[:300]}...")
else:
    st.info("ℹ️ Add some news URLs above to start building your knowledge base.")
