# Interactive Retrieval-Augmented Generation (RAG) with Graph-Augmented Reasoning

An interactive Retrieval-Augmented Generation (RAG) system for question answering over live news articles and official reports.  
Originally built with FAISS, LangChain, OpenAI, and Streamlit, the system is now extended with **graph-based reasoning** using Graph Neural Networks (GNNs) for improved retrieval quality and factual grounding.

---

## ✨ Features

- **URL Ingestion**: Fetch and index news articles dynamically.  
- **Vector Database (FAISS)**: Store and retrieve semantically relevant chunks.  
- **LLM-Powered Answers**: Use OpenAI API to generate concise responses grounded in retrieved content.  
- **Streamlit App**: Simple interface for real-time querying with source links.  

### 🔹 New: Graph-Augmented Retrieval
- Extracts entities (people, organizations, locations) from articles using **spaCy**.  
- Builds a **document–entity graph** with **NetworkX**.  
- Learns graph embeddings with a lightweight **GraphSAGE model** (PyTorch Geometric).  
- Combines FAISS similarity scores with GNN-based reasoning for hybrid retrieval.  
- Improves factual consistency, reduces hallucinations, and surfaces articles connected via shared entities (even if wording differs).  

---

## 🛠 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/graph-rag.git
cd graph-rag

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
