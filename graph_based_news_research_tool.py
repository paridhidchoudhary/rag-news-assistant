import os
import streamlit as st
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup

import spacy
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GraphSAGE
from sklearn.metrics.pairwise import cosine_similarity

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
st.title("📰 Graph-Augmented News Research Assistant")

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
# Graph + GNN Utilities
# -----------------------
nlp = spacy.load("en_core_web_sm")

def build_graph_from_texts(texts):
    G = nx.Graph()
    for i, doc in enumerate(texts):
        doc_id = f"doc_{i}"
        G.add_node(doc_id, type="doc", text=doc)
        spacy_doc = nlp(doc[:1000])  # limit for speed
        for ent in spacy_doc.ents:
            ent_node = f"ent_{ent.text}"
            if not G.has_node(ent_node):
                G.add_node(ent_node, type="entity")
            G.add_edge(doc_id, ent_node)
    return G

# class SimpleGraphSAGE(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super().__init__()
#         self.conv1 = GraphSAGE(in_channels, hidden_channels)
#         self.conv2 = GraphSAGE(hidden_channels, out_channels)

#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index).relu()
#         x = self.conv2(x, edge_index)
#         return x

class SimpleGraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GraphSAGE(in_channels, hidden_channels, num_layers=1)
        self.conv2 = GraphSAGE(hidden_channels, out_channels, num_layers=1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

def graph_to_embeddings(G):
    nodes = list(G.nodes)
    node_index = {n: i for i, n in enumerate(nodes)}
    edge_index = torch.tensor([[node_index[u], node_index[v]] for u, v in G.edges]).t().contiguous()

    # simple identity features
    x = torch.eye(len(nodes))
    data = Data(x=x, edge_index=edge_index)

    # Change out_channels to 1536 to match OpenAI embeddings
    model = SimpleGraphSAGE(in_channels=len(nodes), hidden_channels=64, out_channels=1536)
    with torch.no_grad():
        out = model(data.x, data.edge_index)

    return {n: out[i].numpy() for n, i in node_index.items() if G.nodes[n]["type"] == "doc"}
def hybrid_retrieve(query_emb, faiss_store, gnn_embs, alpha=0.7, k=3):
    # FAISS similarity
    docs = faiss_store.similarity_search_by_vector(query_emb, k=k)
    faiss_scores = {d.page_content: 1.0 for d in docs}

    # GNN similarity
    gnn_scores = {}
    for doc, emb in gnn_embs.items():
        sim = cosine_similarity([query_emb], [emb])[0][0]
        gnn_scores[doc] = sim

    # Combine scores
    combined = {}
    for doc in faiss_scores:
        combined[doc] = alpha * faiss_scores.get(doc, 0) + (1 - alpha) * gnn_scores.get(doc, 0)

    # Sort and return top k
    sorted_docs = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:k]
    return [d[0] for d in sorted_docs]

# -----------------------
# Upload URLs
# -----------------------
st.subheader("🔗 Add News Articles")
urls_input = st.text_area("Enter one or more URLs (comma or newline separated):")

# if st.button("Process URLs"):
#     if urls_input.strip():
#         urls = [u.strip() for u in urls_input.replace(",", "\n").split("\n") if u.strip()]
#         texts = [extract_text_from_url(u) for u in urls]

#         if vectorstore is None:
#             vectorstore = FAISS.from_texts(texts, embedding=embeddings)
#         else:
#             vectorstore.add_texts(texts)

#         vectorstore.save_local(VECTORSTORE_DIR)
#         st.success(f"✅ Added {len(texts)} documents to the index.")

#         # Build graph + embeddings
#         st.info("📊 Building graph and GNN embeddings...")
#         global_graph = build_graph_from_texts(texts)
#         global_gnn_embs = graph_to_embeddings(global_graph)

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

        # Build graph + embeddings
        st.info("📊 Building graph and GNN embeddings...")
        global_graph = build_graph_from_texts(texts)
        global_gnn_embs = graph_to_embeddings(global_graph)
        st.session_state["global_gnn_embs"] = global_gnn_embs  # <-- Save to session_state

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

    # if query:
    #     with st.spinner("Thinking..."):
    #         q_emb = embeddings.embed_query(query)
    #         # Hybrid retrieval (FAISS + GNN)
    #         hybrid_docs = hybrid_retrieve(q_emb, vectorstore, global_gnn_embs, alpha=0.7, k=3)
    #         result = qa_chain({"question": query})

    #     st.markdown("### 📌 Answer")
    #     st.write(result["answer"])

    #     st.markdown("### 📚 Hybrid Sources")
    #     for i, doc in enumerate(hybrid_docs, 1):
    #         st.write(f"**Doc {i}:** {doc[:300]}...")

    if query:
        with st.spinner("Thinking..."):
            q_emb = embeddings.embed_query(query)
            # Retrieve GNN embeddings from session_state
            global_gnn_embs = st.session_state.get("global_gnn_embs", {})
            hybrid_docs = hybrid_retrieve(q_emb, vectorstore, global_gnn_embs, alpha=0.7, k=3)
            result = qa_chain({"question": query})

        st.markdown("### 📌 Answer")
        st.write(result["answer"])

        st.markdown("### 📚 Hybrid Sources")
        for i, doc in enumerate(hybrid_docs, 1):
            st.write(f"**Doc {i}:** {doc[:300]}...")

else:
    st.info("ℹ️ Add some news URLs above to start building your knowledge base.")
