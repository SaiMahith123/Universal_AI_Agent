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
        .file-card { padding: 12px; border-radius: 10px; border: 1px solid #4A4A4A; background-color: #1E1E1E; display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
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
if "image_list" not in st.session_state: st.session_state.image_list = []

@st.cache_resource
def get_embed_model(): return SentenceTransformer('all-MiniLM-L6-v2')

# --- DOC PROCESSING (PDF & PPT) ---
def process_docs(uploaded_file):
    chunks, metadata = [], []
    if uploaded_file.name.lower().endswith('.pdf'):
        reader = PyPDF2.PdfReader(uploaded_file)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                p_chunks = [text[j:j+1000] for j in range(0, len(text), 800)]
                chunks.extend(p_chunks)
                metadata.extend([f"PDF Pg {i+1}"] * len(p_chunks))
    elif uploaded_file.name.lower().endswith(('.ppt', '.pptx')):
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
    model_choice = st.selectbox("Select Brain:", ["llama-3.3-70b-versatile", "meta-llama/llama-4-scout-17b-16e-instruct", "deepseek-r1-distill-llama-70b"])
    st.info("💡 **Llama 3.3:** Docs | **Llama 4:** Images | **DeepSeek:** Logic")
    voice_on = st.toggle("🔊 Auto-Play AI Voice", value=False)
    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.vector_store = None
        st.session_state.image_list = []
        st.rerun()
    st.caption("Developed by T Sai Mahit | B.E in AI-ML (2021-25)")

# DISPLAY CHAT HISTORY
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if "files" in msg:
            for f_info in msg["files"]:
                st.markdown(f"""<div class="file-card">📄 <b>{f_info['name']}</b></div>""", unsafe_allow_html=True)
                if f_info['type'] == "Image":
                    st.image(base64.b64decode(f_info['data']), width=250)
        st.markdown(msg["content"])

# --- INPUT & MULTIMODAL LOGIC ---
prompt_data = st.chat_input("Ask, Upload Docs/Images...", accept_file=True, accept_audio=True)

if prompt_data:
    user_text = prompt_data.text or ""
    current_files = []

    # 1. Process All Uploaded Files
    if prompt_data.files:
        for f in prompt_data.files
