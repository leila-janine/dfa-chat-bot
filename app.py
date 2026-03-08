import gradio as gr
import faiss
import json
import os
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

# ==========================================
# 1. INITIALIZE MODELS & API
# ==========================================
# Read the secret token from the environment (we will set this in the cloud)
hf_token = os.environ.get("HF_TOKEN")
client = InferenceClient(token=hf_token)

# Load lightweight embedding model
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load database files
try:
    with open("dfa_chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    index = faiss.read_index("dfa_faiss.index")
    db_loaded = True
except Exception as e:
    db_loaded = False
    error_msg = str(e)

# ==========================================
# 2. RETRIEVAL LOGIC
# ==========================================
def retrieve(query, k=3):
    q_emb = embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, idxs = index.search(q_emb, k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        c = chunks[int(idx)]
        results.append({**c, "score": float(score)})
    return results

# ==========================================
# 3. GENERATION & CHAT LOGIC
# ==========================================
def respond(message, history):
    if not db_loaded:
        return f"🚨 Database failed to load. Please check your files. Error: {error_msg}"
        
    if not message.strip():
        return "⚠️ Please enter a valid question."
    
    try:
        # 1. Retrieve Documents
        hits = retrieve(message, k=3)
        MIN_SCORE = 0.45 
        if not hits or hits[0]["score"] < MIN_SCORE:
            return "I don't know based on the provided PDFs. The requested information is not in the DFA manual."
        
        # 2. Format Context
        context_text = "\n\n".join(
            [f"[{i+1}] (Source: {h['source']}, page {h['page']}) {h['text']}" for i, h in enumerate(hits)]
        )
        
        # 3. Strict Phi-3 Prompt Template
        prompt = f"""<|system|>
You are a strict document-based question-answering system. Use ONLY the context provided to answer.
Provide a direct, concise answer. If the info is missing, say you don't know.
STOP immediately after answering.<|end|>
<|user|>
Context:
{context_text}

Question: {message}<|end|>
<|assistant|>"""
              
        # 4. Call Hugging Face API
        response = client.text_generation(
            prompt, 
            model="microsoft/Phi-3-mini-4k-instruct",
            max_new_tokens=250, 
            return_full_text=False
        )
        
        # 5. Clean up Answer
        answer = response.strip()
        answer = answer.split("Question:")[0].split("<|user|>")[0].split("Answer:")[0].strip()
        
        # 6. Append Context for Grading Rubric
        context_display = "\n\n---\n**🔍 Retrieved Context & Citations:**\n"
        for i, h in enumerate(hits, 1):
            context_display += f"* **[{i}] {h['source']} (Page {h['page']})** | Similarity: {h['score']:.3f}\n> {h['text']}\n\n"
            
        return answer + context_display
        
    except Exception as e:
        return f"❌ An API or connection error occurred: {str(e)}"

# ==========================================
# 4. BUILD GRADIO UI
# ==========================================
demo = gr.ChatInterface(
    fn=respond,
    title="📜 DFA Operations Manual Chatbot",
    description="Ask procedural and policy questions based on the DFA Authentication Division Operations Manual.",
    theme="soft"
)

if __name__ == "__main__":
    demo.launch()