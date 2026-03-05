import streamlit as st
from groq import Groq
import pypdf as PyPDF2
import base64

st.set_page_config(
    page_title="Mahi's Universal Groq AI", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 Mahi's Universal Groq AI")

if "GROQ_API_KEY" in st.secrets:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
else:
    st.error("❌ GROQ_API_KEY not found in Streamlit Secrets!")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "file_data" not in st.session_state:
    st.session_state.file_data = {"text": "", "image_b64": None, "type": None}

with st.sidebar:
    st.write("📌 NOTE")
    st.info("""
    **Model Selection Guide:**
    * **Llama 3.3 70B:** General Logic & Chat.
    * **Llama 3.2 Vision (11B):** Required for Images.
    * **DeepSeek R1:** Complex Reasoning & Math.
    * **Mixtral 8x7B:** Large PDF Summarization.
    """)
    
    st.divider()
    model_choice = st.selectbox(
        "Select AI Brain:",
        [
            "llama-3.3-70b-versatile", 
            "llama-3.2-11b-vision-preview", 
            "mixtral-8x7b-32768", 
            "deepseek-r1-distill-llama-70b"
        ]
    )
    
    if st.session_state.messages:
        chat_history = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages])
        st.download_button("📥 Save Chat History", chat_history, file_name="mahi_ai_chat.txt")

    st.divider()
    st.caption("Developed by T Sai Mahit | B.E in AI-ML (2021-25)")

def process_file(file):
    if file.type == "application/pdf":
        reader = PyPDF2.PdfReader(file)
        full_text = "".join([page.extract_text() for page in reader.pages])
        return {"text": full_text[:10000], "image_b64": None, "type": "pdf"}
    elif "image" in file.type:
        b64_image = base64.b64encode(file.getvalue()).decode('utf-8')
        return {"text": "", "image_b64": b64_image, "type": "image"}
    return None

uploaded_file = st.file_uploader("Upload PDF or Image", type=["pdf", "jpg", "jpeg", "png"])

if uploaded_file:
    if "image" in uploaded_file.type and "vision" not in model_choice:
        st.warning("⚠️ Warning: Please select the 'llama-3.2-11b-vision-preview' model for images.")
    
    with st.spinner("Processing file..."):
        st.session_state.file_data = process_file(uploaded_file)
        st.success(f"Loaded: {uploaded_file.name}")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about your file..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.file_data["image_b64"] and "vision" in model_choice:
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{st.session_state.file_data['image_b64']}"}}
            ]
        }]
    else:
        context = st.session_state.file_data["text"]
        messages = [{"role": "user", "content": f"Context: {context}\n\nQuestion: {prompt}"}]

    with st.chat_message("assistant"):
        try:
            response_stream = client.chat.completions.create(
                model=model_choice,
                messages=messages,
                stream=True
            )
            
            def parse_stream(stream):
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content

            full_response = st.write_stream(parse_stream(response_stream))
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        except Exception as e:
            st.error(f"❌ AI Error: {str(e)}")
