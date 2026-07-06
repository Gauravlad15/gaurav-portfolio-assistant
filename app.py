import streamlit as st
import joblib
import re
import numpy as np
import os
from numpy import float32
from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq

# -------------------------------------------------------------------
# 1. PAGE CONFIG
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Gaurav's AI Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------------
# 2. CUSTOM CSS
# -------------------------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #0f1117 0%, #161925 100%);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #7C3AED, #22D3EE);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .hero-caption {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-top: -8px;
    }
    [data-testid="stChatMessage"] {
        border-radius: 16px;
        padding: 10px 16px;
        margin-bottom: 10px;
        border: 1px solid rgba(255,255,255,0.06);
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background: linear-gradient(135deg, #312e81, #1e1b4b);
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        background: rgba(255,255,255,0.04);
    }
    [data-testid="stChatInput"] {
        border-radius: 14px;
        border: 1px solid rgba(124,58,237,0.4);
    }
    [data-testid="stSidebar"] {
        background: #12141c;
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-thumb { background: #4c1d95; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# 3. TOKENIZER 
# -------------------------------------------------------------------
def _tokenize(text):
    return re.findall(r'\w+', text.lower())

# -------------------------------------------------------------------
# 4. LOAD CONFIG & MODELS 
# -------------------------------------------------------------------
@st.cache_resource
   

def load_resources():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    api_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
    if not api_key:
        st.error("GROQ_API_KEY not found. Set it in HF Space secrets or .streamlit/secrets.toml locally.")
        st.stop()

    groq_client = Groq(api_key=api_key)

    try:
        split_docs = joblib.load('split_docs.pkl')
        index = joblib.load('index.pkl')
        bm25 = joblib.load('bm25.pkl')
    except FileNotFoundError:
        return None, None, None, model, cross_encoder, groq_client

    return index, split_docs, bm25, model, cross_encoder, groq_client

index, split_docs, bm25, model, cross_encoder, groq_client = load_resources()

# -------------------------------------------------------------------
# 5. HELPER FUNCTIONS — Hybrid Search + Re-ranking
# -------------------------------------------------------------------
def retrieval(query, model, index, split_docs, bm25, cross_encoder, k=4, fetch_k=15):
    # Vector search (FAISS)
    query_embeddings = np.array(model.encode([query])).astype(float32)
    _, indices = index.search(query_embeddings, fetch_k)
    vector_indices = [i for i in indices[0] if i != -1]

    # BM25 keyword search
    bm25_scores = bm25.get_scores(_tokenize(query))
    bm25_indices = list(np.argsort(bm25_scores)[::-1][:fetch_k])

    # Combine + dedupe
    candidate_indices = list(dict.fromkeys(vector_indices + bm25_indices))

    # Cross-encoder re-ranking
    pairs = [[query, split_docs[i].page_content] for i in candidate_indices]
    rerank_scores = cross_encoder.predict(pairs)

    ranked = sorted(zip(candidate_indices, rerank_scores), key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, _ in ranked[:k]]

    results = [split_docs[i].page_content for i in top_indices]
    return results

def generate_answer(query, results, groq_client):
    context = "\n\n".join(results)
    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.
Synthesize information across ALL provided sections if relevant (e.g. combine Experience + Goals for career questions).
If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {query}

Answer:"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1000
    )
    return response.choices[0].message.content

# -------------------------------------------------------------------
# 6. SIDEBAR
# -------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🤖 About this bot")
    st.markdown("Ask me anything about **Gaurav's** background, skills, and projects.")
    st.divider()
    st.markdown("**Try asking:**")
    st.markdown("- What are his technical skills?\n- What's his CGPA?\n- Is he job ready?\n- What projects has he built?")
    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# -------------------------------------------------------------------
# 7. MAIN UI
# -------------------------------------------------------------------
st.markdown('<p class="hero-title">🤖 Gaurav\'s AI Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-caption">Ask me anything — powered by RAG over his profile data</p>', unsafe_allow_html=True)
st.write("")

if index is None or split_docs is None or bm25 is None:
    st.error("⚠️ 'split_docs.pkl', 'index.pkl' ya 'bm25.pkl' nahi mili! Pehle apni Notebook ka aakhri cell run karke unhe generate karo.")
else:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    if query := st.chat_input("Ask something about Gaurav..."):
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Searching database..."):
                results = retrieval(query, model, index, split_docs, bm25, cross_encoder, k=4)
                answer = generate_answer(query, results, groq_client)
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})