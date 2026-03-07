import streamlit as st
from groq import Groq
import pypdf as PyPDF2
from pptx import Presentation
import base64
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import tempfile
import os
from streamlit_TTS import auto_play, text_to_audio

# --- CONFIG & STYLING ---
st.set_page_config(page_title="Mahi's Universal Groq AI", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
        .stApp { flex-direction: row-reverse; }
        [data-testid="stSidebar"] { left: auto !important; right: 0 !important; border-left: 1px solid rgba(250, 250, 250, 0.1); }
        .stChatInputContainer { position: fixed; bottom: 20px; z-index: 1000; }
        .main .block-container { padding-bottom: 150px; }
        .file-card { padding: 10px; border-radius: 10px; border: 1px solid #4A4A4A; background-color: #1E1E1E; display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 Mahi's Universal Groq AI")

# API Setup
if "GROQ_API_KEY" in st.secrets:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
else:
    st.error("❌ GROQ_API_KEY missing!")
    st.stop()

# PERSISTENT STATE
if "messages" not in st.session_state: st.session_state.messages = []
if "vector_store" not in st.session_state: st.session_state.vector_store = None
if "file_context" not in st.session_state: st.session_state.file_context = {"image_b64": None}

@st.cache_resource
def get_embed_model(): return SentenceTransformer('all-MiniLM-L6-v2')

# --- DOC PROCESSING ---
def process_docs(uploaded_file):
    chunks, metadata = [], []
    if uploaded_file.name.endswith('.pdf'):
        reader = PyPDF2.PdfReader(uploaded_file)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                p_chunks = [text[j:j+1000] for j in range(0, len(text), 800)]
                chunks.extend(p_chunks)
                metadata.extend([f"PDF Pg {i+1}"] * len(p_chunks))
    elif uploaded_file.name.endswith(('.ppt', '.pptx')):
        prs = Presentation(uploaded_file)
        for i, slide in enumerate(prs.slides):
            text = "".join([s.text + " " for s in slide.shapes if hasattr(s, "text")])
            if text:
                s_chunks = [text[j:j+1000] for j in range(0, len(text), 800)]
                chunks.extend(s_chunks)
                metadata.extend([f"PPT Slide {i+1}"] * len(s_chunks))
    
    if chunks:
        em = get_embed_model()
        embeddings = em.encode(chunks)
        idx = faiss.IndexFlatL2(embeddings.shape[1])
        idx.add(np.array(embeddings).astype('float32'))
        st.session_state.vector_store = {"index": idx, "chunks": chunks, "sources": metadata}
        return True
    return False

# --- SIDEBAR UI ---
with st.sidebar:
    st.header("⚙️ Settings")
    # CHANGING MODEL NO LONGER CLEARS CHAT because state is handled separately
    model_choice = st.selectbox("Select Brain:", ["llama-3.3-70b-versatile", "meta-llama/llama-4-scout-17b-16e-instruct", "deepseek-r1-distill-llama-70b"])
    st.info("💡 **Llama 3.3:** Docs | **Llama 4:** Images | **DeepSeek:** Logic")
    voice_on = st.toggle("🔊 Auto-Play AI Voice", value=False)
    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.vector_store = None
        st.rerun()
    st.caption("Developed by T Sai Mahit | B.E in AI-ML (2021-25)")

# DISPLAY CHAT HISTORY WITH FILE PREVIEWS
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if "file_info" in msg:
            st.markdown(f"""<div class="file-card">📄 <b>{msg['file_info']['name']}</b> ({msg['file_info']['type']})</div>""", unsafe_allow_html=True)
            if msg['file_info']['type'] == "Image":
                st.image(base64.b64decode(msg['file_info']['data']), width=300)
        st.markdown(msg["content"])

# --- INPUT & LOGIC ---
prompt_data = st.chat_input("Ask, Upload PDF/PPT/Image...", accept_file=True, accept_audio=True)

if prompt_data:
    user_text = prompt_data.text or ""
    file_msg = None

    # Process Uploads and Create Visual Entry
    if prompt_data.files:
        f = prompt_data.files[0]
        if f.name.endswith(('.pdf', '.pptx', '.ppt')):
            process_docs(f)
            file_msg = {"role": "user", "content": f"I've uploaded a file: {f.name}", "file_info": {"name": f.name, "type": "Document"}}
        elif "image" in f.type:
            img_b64 = base64.b64encode(f.getvalue()).decode('utf-8')
            st.session_state.file_context["image_b64"] = img_b64
            file_msg = {"role": "user", "content": f"I've uploaded an image: {f.name}", "file_info": {"name": f.name, "type": "Image", "data": img_b64}}

    if file_msg: st.session_state.messages.append(file_msg)
    if user_text: st.session_state.messages.append({"role": "user", "content": user_text})
    
    # RAG + MEMORY BUILD
    context = ""
    if st.session_state.vector_store and user_text:
        qv = get_embed_model().encode([user_text])
        D, I = st.session_state.vector_store["index"].search(np.array(qv).astype('float32'), k=5)
        context = "\n".join([st.session_state.vector_store["chunks"][i] for i in I[0]])

    api_messages = [{"role": "system", "content": "You are Mahi's AI. Use context and history to answer."}]
    for m in st.session_state.messages[-10:]: # Sliding window memory
        api_messages.append({"role": m["role"], "content": m["content"]})

    # PROMPT ROUTING
    active_model = "meta-llama/llama-4-scout-17b-16e-instruct" if st.session_state.file_context["image_b64"] else model_choice
    final_prompt = f"Context:\n{context}\n\nUser Question: {user_text}" if context else user_text
    
    # GENERATE
    with st.chat_message("assistant"):
        try:
            stream = client.chat.completions.create(model=active_model, messages=api_messages, stream=True)
            resp = st.write_stream(c.choices[0].delta.content for c in stream if c.choices[0].delta.content)
            st.session_state.messages.append({"role": "assistant", "content": resp})
            st.session_state.file_context["image_b64"] = None
            if voice_on: auto_play(text_to_audio(resp))
        except Exception as e: st.error(f"Error: {str(e)}")
    st.rerun()
