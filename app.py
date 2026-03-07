import streamlit as st
from groq import Groq
import pypdf as PyPDF2
import base64
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import tempfile
import os

st.set_page_config(page_title="Mahi's Universal Groq AI", layout="wide")
st.title("🤖 Mahi's Universal Groq AI")

if "GROQ_API_KEY" in st.secrets:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
else:
    st.error("❌ GROQ_API_KEY missing in Secrets!")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "file_context" not in st.session_state:
    st.session_state.file_context = {"text": "", "image_b64": None}

@st.cache_resource
def get_embed_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def process_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        reader = PyPDF2.PdfReader(uploaded_file)
        text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
        embed_model = get_embed_model()
        embeddings = embed_model.encode(chunks)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings).astype('float32'))
        st.session_state.vector_store = {"index": index, "chunks": chunks}
        return {"text": text[:2000], "image_b64": None}
    elif "image" in uploaded_file.type:
        b64 = base64.b64encode(uploaded_file.getvalue()).decode('utf-8')
        return {"text": "", "image_b64": b64}
    return None

with st.sidebar:
    st.header("🧠 AI Brain & Guide")
  
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        model_choice = st.radio(
            "Select Model:",
            ["llama-3.3-70b-versatile", "meta-llama/llama-4-scout-17b-16e-instruct", "deepseek-r1-distill-llama-70b"],
            help="Choose the specialized brain for your task."
        )
    
    with col2:
        st.markdown("""
        **Quick Guide:**
        * **Llama 3.3:** Best for PDFs.
        * **Llama 4 Scout:** Best for Images.
        * **DeepSeek R1:** Best for Logic.
        """)

    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.vector_store = None
        st.rerun()
    st.caption("Developed by T Sai Mahit | B.E in AI-ML (2021-25)")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt_data = st.chat_input(
    "Ask, Upload, or Record...", 
    accept_file=True, 
    file_type=["pdf", "jpg", "png", "jpeg"], 
    accept_audio=True
)

if prompt_data:
    user_text = prompt_data.text or ""
    
    if prompt_data.audio:
        with st.spinner("🎤 Transcribing your voice..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
                tmp_audio.write(prompt_data.audio.getvalue())
                tmp_audio_path = tmp_audio.name
            with open(tmp_audio_path, "rb") as af:
                tr = client.audio.transcriptions.create(file=af, model="whisper-large-v3")
                user_text = tr.text
            os.remove(tmp_audio_path)

    if prompt_data.files:
        with st.spinner("📂 Indexing file for RAG..."):
            st.session_state.file_context = process_file(prompt_data.files[0])

    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    context = ""
    if st.session_state.vector_store and user_text:
        em = get_embed_model()
        qv = em.encode([user_text])
        D, I = st.session_state.vector_store["index"].search(np.array(qv).astype('float32'), k=3)
        context = "\n".join([st.session_state.vector_store["chunks"][i] for i in I[0]])

    active_model = model_choice
    if st.session_state.file_context["image_b64"]:
        active_model = "meta-llama/llama-4-scout-17b-16e-instruct"
        msgs = [{"role": "user", "content": [{"type": "text", "text": user_text}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{st.session_state.file_context['image_b64']}"}}]}]
    else:
        full_p = f"Context: {context}\n\nUser: {user_text}" if context else user_text
        msgs = [{"role": "user", "content": full_p}]

    with st.chat_message("assistant"):
        try:
            stream = client.chat.completions.create(model=active_model, messages=msgs, stream=True)
            def parse(s):
                for c in s:
                    if c.choices[0].delta.content: yield c.choices[0].delta.content
            resp = st.write_stream(parse(stream))
            st.session_state.messages.append({"role": "assistant", "content": resp})
        except Exception as e:
            st.error(f"Error: {str(e)}")
