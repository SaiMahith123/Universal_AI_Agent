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

# API Initialization
if "GROQ_API_KEY" in st.secrets:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
else:
    st.error("❌ GROQ_API_KEY missing in Secrets!")
    st.stop()

# PERSISTENT STATE
if "messages" not in st.session_state: st.session_state.messages = []
if "vector_store" not in st.session_state: st.session_state.vector_store = None
if "file_context" not in st.session_state: st.session_state.file_context = {"image_b64": None}

@st.cache_resource
def get_embed_model(): return SentenceTransformer('all-MiniLM-L6-v2')

# --- DOC PROCESSING (PDF & PPT) ---
def process_docs(uploaded_file):
    chunks, metadata = [], []
    try:
        if uploaded_file.name.endswith('.pdf'):
            reader = PyPDF2.PdfReader(uploaded_file)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    # Recursive overlap for better context
                    p_chunks = [text[j:j+1000] for j in range(0, len(text), 800)]
                    chunks.extend(p_chunks)
                    metadata.extend([f"PDF Pg {i+1}"] * len(p_chunks))
        
        elif uploaded_file.name.endswith(('.ppt', '.pptx')):
            prs = Presentation(uploaded_file)
            for i, slide in enumerate(prs.slides):
                text = " ".join([shape.text for shape in slide.shapes if hasattr(shape, "text")])
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
    except Exception as e:
        st.error(f"Processing Error: {e}")
    return False

# --- SIDEBAR UI ---
with st.sidebar:
    st.header("⚙️ Settings")
    # State-safe model selection
    model_choice = st.selectbox("Select Brain:", ["llama-3.3-70b-versatile", "meta-llama/llama-4-scout-17b-16e-instruct", "deepseek-r1-distill-llama-70b"])
    st.info("💡 **Llama 3.3:** PDFs/PPTs | **Llama 4:** Images | **DeepSeek:** Logic")
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
    
    # 1. Process Files and Create Visual History
    if prompt_data.files:
        f = prompt_data.files[0]
        if f.name.lower().endswith(('.pdf', '.pptx', '.ppt')):
            process_docs(f)
            st.session_state.messages.append({"role": "user", "content": f"I've uploaded a file: {f.name}", "file_info": {"name": f.name, "type": "Document"}})
        elif any(ext in f.type for ext in ["image/png", "image/jpeg"]):
            img_b64 = base64.b64encode(f.getvalue()).decode('utf-8')
            st.session_state.file_context["image_b64"] = img_b64
            st.session_state.messages.append({"role": "user", "content": f"I've uploaded an image: {f.name}", "file_info": {"name": f.name, "type": "Image", "data": img_b64}})

    if user_text:
        st.session_state.messages.append({"role": "user", "content": user_text})
        st.chat_message("user").markdown(user_text)

    # 2. RAG Retrieval
    context = ""
    if st.session_state.vector_store and user_text:
        qv = get_embed_model().encode([user_text])
        D, I = st.session_state.vector_store["index"].search(np.array(qv).astype('float32'), k=5)
        context = "\n".join([st.session_state.vector_store["chunks"][i] for i in I[0]])

    # 3. BUILD MEMORY-AWARE API CALL
    api_messages = [{"role": "system", "content": "You are Mahi's Universal AI. ALWAYS use the provided PDF/PPT context to answer if available. Remember the previous conversation turns."}]
    
    # Send last 8 messages for robust memory
    for m in st.session_state.messages[-8:]:
        api_messages.append({"role": m["role"], "content": m["content"]})

    # Add context to the LATEST message
    if context and api_messages:
        api_messages[-1]["content"] = f"Context from Files:\n{context}\n\nQuestion: {api_messages[-1]['content']}"

    # 4. GENERATE RESPONSE
    active_model = "meta-llama/llama-4-scout-17b-16e-instruct" if st.session_state.file_context["image_b64"] else model_choice
    
    with st.chat_message("assistant"):
        try:
            stream = client.chat.completions.create(model=active_model, messages=api_messages, stream=True)
            full_resp = st.write_stream(c.choices[0].delta.content for c in stream if c.choices[0].delta.content)
            st.session_state.messages.append({"role": "assistant", "content": full_resp})
            st.session_state.file_context["image_b64"] = None
            if voice_on: auto_play(text_to_audio(full_resp))
        except Exception as e:
            st.error(f"Error: {str(e)}")
    st.rerun()
