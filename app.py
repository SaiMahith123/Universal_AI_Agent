import streamlit as st
from groq import Groq
import pypdf as PyPDF2
import base64
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import tempfile
import os
from streamlit_TTS import auto_play, text_to_audio

# --- CONFIGURATION & STYLING ---
st.set_page_config(page_title="Mahi's Universal Groq AI", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
        .stApp { flex-direction: row-reverse; }
        [data-testid="stSidebar"] { left: auto !important; right: 0 !important; border-left: 1px solid rgba(250, 250, 250, 0.1); }
        .stChatInputContainer { position: fixed; bottom: 20px; z-index: 1000; }
        .main .block-container { padding-bottom: 150px; }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 Mahi's Universal Groq AI")

# API Initialization
if "GROQ_API_KEY" in st.secrets:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
else:
    st.error("❌ GROQ_API_KEY missing in Secrets!")
    st.stop()

# State Management
if "messages" not in st.session_state: st.session_state.messages = []
if "vector_store" not in st.session_state: st.session_state.vector_store = None
if "file_context" not in st.session_state: st.session_state.file_context = {"image_b64": None}

@st.cache_resource
def get_embed_model(): 
    return SentenceTransformer('all-MiniLM-L6-v2')

# --- PDF PROCESSING LOGIC ---
def process_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    chunks = []
    metadata = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            # Recursive overlap (1000 size, 200 overlap) for large docs
            page_chunks = [text[j:j+1000] for j in range(0, len(text), 800)]
            chunks.extend(page_chunks)
            metadata.extend([i + 1] * len(page_chunks)) 
            
    if chunks:
        em = get_embed_model()
        embeddings = em.encode(chunks)
        idx = faiss.IndexFlatL2(embeddings.shape[1])
        idx.add(np.array(embeddings).astype('float32'))
        st.session_state.vector_store = {"index": idx, "chunks": chunks, "pages": metadata}
        return True
    return False

# --- SIDEBAR UI ---
with st.sidebar:
    st.header("⚙️ Settings")
    model_choice = st.selectbox("Select Brain:", 
                                ["llama-3.3-70b-versatile", 
                                 "meta-llama/llama-4-scout-17b-16e-instruct", 
                                 "deepseek-r1-distill-llama-70b"])
    
    st.info("💡 **Llama 3.3:** PDFs | **Llama 4:** Images | **DeepSeek:** Logic")
    voice_on = st.toggle("🔊 Auto-Play AI Voice", value=False)
    
    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.vector_store = None
        st.session_state.file_context["image_b64"] = None
        st.rerun()
        
    st.caption("Developed by T Sai Mahit | B.E in AI-ML (2021-25)")

# --- CHAT DISPLAY ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): 
        st.markdown(msg["content"])

# --- INPUT HANDLING ---
prompt_data = st.chat_input("Ask, Upload PDF/Image, or Record...", accept_file=True, accept_audio=True)

if prompt_data:
    user_text = prompt_data.text or ""
    
    # Audio Processing (Whisper)
    if prompt_data.audio:
        with st.spinner("🎤 Transcribing..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(prompt_data.audio.getvalue())
                tmp_path = tmp.name
            with open(tmp_path, "rb") as af:
                tr = client.audio.transcriptions.create(file=af, model="whisper-large-v3")
                user_text = tr.text
            os.remove(tmp_path)

    # File Processing
    if prompt_data.files:
        for f in prompt_data.files:
            if f.type == "application/pdf":
                with st.spinner("📂 Indexing PDF..."): 
                    process_pdf(f)
            elif "image" in f.type:
                st.session_state.file_context["image_b64"] = base64.b64encode(f.getvalue()).decode('utf-8')

    if user_text or st.session_state.file_context["image_b64"]:
        # Show User Message
        with st.chat_message("user"): 
            st.markdown(user_text)

        # RAG Retrieval
        context_text = ""
        sources = []
        if st.session_state.vector_store and user_text:
            qv = get_embed_model().encode([user_text])
            D, I = st.session_state.vector_store["index"].search(np.array(qv).astype('float32'), k=5)
            for i in I[0]:
                context_text += st.session_state.vector_store["chunks"][i] + "\n"
                sources.append(st.session_state.vector_store["pages"][i])

        # --- CONVERSATION MEMORY LOGIC ---
        # We start with a system prompt and append the last 6 messages
        history_msgs = [{"role": "system", "content": "You are Mahi's AI. Use the provided context and history to answer accurately."}]
        
        # Add past interactions to give the AI "Memory"
        for m in st.session_state.messages[-6:]: 
            history_msgs.append({"role": m["role"], "content": m["content"]})

        # Set active model and current content
        active_model = model_choice
        if st.session_state.file_context["image_b64"]:
            active_model = "meta-llama/llama-4-scout-17b-16e-instruct"
            current_prompt = [
                {"type": "text", "text": user_text if user_text else "Analyze this image."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{st.session_state.file_context['image_b64']}"}}
            ]
        else:
            page_info = f"(Found on Pages: {list(set(sources))})" if sources else ""
            current_prompt = f"PDF Context:\n{context_text}\n\nUser Question: {user_text}\n{page_info}"

        # Add current user prompt to the list for the API
        history_msgs.append({"role": "user", "content": current_prompt})
        
        # Finally, save the user text to state for display
        st.session_state.messages.append({"role": "user", "content": user_text})

        # --- ASSISTANT RESPONSE ---
        with st.chat_message("assistant"):
            try:
                stream = client.chat.completions.create(model=active_model, messages=history_msgs, stream=True)
                
                def parse(s):
                    for c in s:
                        if c.choices[0].delta.content: yield c.choices[0].delta.content
                
                full_resp = st.write_stream(parse(stream))
                st.session_state.messages.append({"role": "assistant", "content": full_resp})
                
                # Reset image context so it doesn't linger
                st.session_state.file_context["image_b64"] = None
                
                if voice_on:
                    auto_play(text_to_audio(full_resp, language='en'))
            except Exception as e:
                st.error(f"Error: {str(e)}")
