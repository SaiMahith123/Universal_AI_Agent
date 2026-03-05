import streamlit as st
from groq import Groq
import PyPDF2

# Updated the app name in the configuration and title
st.set_page_config(page_title="Mahi's Universal Groq AI", layout="wide")
st.title("🤖 Mahi's Universal Groq AI")

if "GROQ_API_KEY" in st.secrets:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
else:
    st.error("Please add GROQ_API_KEY to Streamlit Secrets!")
    st.stop()

with st.sidebar:
    # Dedicated NOTE section for model selection
    st.header("📌 NOTE")
    st.info("""
    **Select a model based on your task:**
    * **Llama 3.3 70B:** Best for complex logic and general chat.
    * **Llama 3.2 Vision (90B/11B):** Required for Images and Videos.
    * **DeepSeek R1:** Best for Math, Coding, and Deep Thinking.
    * **Mixtral 8x7B:** Best for reading long text from PDFs.
    * **Gemma2 9b:** Best for fast, simple responses.
    """)
    
    st.divider()
    st.header("Model Settings")
    model_choice = st.selectbox(
        "Choose a Model:",
        [
            "llama-3.3-70b-versatile",
            "llama-3.2-90b-vision-preview",
            "llama-3.2-11b-vision-preview",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
            "deepseek-r1-distill-llama-70b"
        ]
    )

uploaded_file = st.file_uploader("Upload PDF, Image, or Video", type=["pdf", "jpg", "jpeg", "png", "mp4"])

def extract_pdf_text(file):
    pdf_reader = PyPDF2.PdfReader(file)
    return "".join([page.extract_text() for page in pdf_reader.pages])

content_context = ""
if uploaded_file:
    if uploaded_file.type == "application/pdf":
        content_context = extract_pdf_text(uploaded_file)
        st.success("PDF Text Extracted!")
    else:
        content_context = f"The user has uploaded a multimedia file: {uploaded_file.name}"
        if "image" in uploaded_file.type:
            st.image(uploaded_file)
        elif "video" in uploaded_file.type:
            st.video(uploaded_file)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask about your file..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    combined_prompt = f"Context: {content_context}\n\nQuestion: {prompt}"
    
    with st.chat_message("assistant"):
        try:
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": combined_prompt}],
                model=model_choice,
            )
            response = chat_completion.choices[0].message.content
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"Error: {e}")
