import streamlit as st
from langchain_groq import ChatGroq
st.set_page_config(page_title="Universal AI", layout="centered")
st.title("🤖 Universal AI Assistant")
if "GROQ_API_KEY" in st.secrets:
    llm = ChatGroq(
        groq_api_key=st.secrets["GROQ_API_KEY"], 
        model_name="llama-3.3-70b-versatile"
    )
else:
    st.error("Missing API Key! Please add GROQ_API_KEY in Streamlit Secrets.")
    st.stop()
if "messages" not in st.session_state:
    st.session_state.messages = []
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
with st.chat_message("assistant"):
    with st.spinner("Processing..."):
        response = llm.invoke(st.session_state.messages)
        st.write(response.content)
        st.session_state.messages.append({"role": "assistant", "content": response.content})
