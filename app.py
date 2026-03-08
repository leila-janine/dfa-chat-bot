import streamlit as st
import faiss
import json
import os
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

# ==========================================
# 1. INITIALIZE MODELS & API
# ==========================================
@st.cache_resource
def load_resources():
    # Streamlit Cloud uses st.secrets instead of os.environ
    hf_token = st.secrets["HF_TOKEN"]
    client = InferenceClient(token=hf_token)
    
    # Small embedding model
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    # Load files
    with open("dfa_chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    index = faiss.read_index("dfa_faiss.index")
    
    return chunks, index, embed_model, client

chunks, index, embed_model, client = load_resources()

# ==========================================
# 2. RAG LOGIC (Same as your Gradio version)
# ==========================================
def retrieve(query, k=3):
    q_emb = embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, idxs = index.search(q_emb, k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        c = chunks[int(idx)]
        results.append({**c, "score": float(score)})
    return results

def generate_answer(message, hits):
    MIN_SCORE = 0.45 
    if not hits or hits[0]["score"] < MIN_SCORE:
        return "I don't know based on the provided PDFs. The requested information is not in the DFA manual.", None
    
    context_text = "\n\n".join(
        [f"[{i+1}] (Source: {h['source']}, page {h['page']}) {h['text']}" for i, h in enumerate(hits)]
    )
    
    prompt = f"""<|system|>
You are a strict document-based question-answering system. Use ONLY the context provided to answer.
Provide a direct, concise answer. If the info is missing, say you don't know.
STOP immediately after answering.<|end|>
<|user|>
Context:
{context_text}

Question: {message}<|end|>
<|assistant|>"""
          
    response = client.text_generation(
        prompt, 
        model="microsoft/Phi-3-mini-4k-instruct",
        max_new_tokens=250, 
        return_full_text=False
    )
    
    answer = response.strip().split("Question:")[0].split("<|user|>")[0].split("<|end|>")[0]
    return answer, hits

# ==========================================
# 3. STREAMLIT UI
# ==========================================
st.set_page_config(page_title="DFA RAG Chatbot", page_icon="📜")
st.title("📜 DFA Operations Manual Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask about the DFA manual..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            hits = retrieve(prompt)
            answer, raw_hits = generate_answer(prompt, hits)
            st.markdown(answer)
            
            # Bonus: Show citations in an expander
            if raw_hits:
                with st.expander("🔍 View Sources"):
                    for h in raw_hits:
                        st.write(f"**{h['source']} (Page {h['page']})** - Score: {h['score']:.3f}")
                        st.caption(h['text'])
            
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            st.error(f"Error: {e}")
