import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="Universal AI", layout="centered")
st.title("🤖 Universal AI Assistant")

if "GROQ_API_KEY" in st.secrets:
    llm = ChatGroq(
        groq_api_key=st.secrets["GROQ_API_KEY"], 
        model_name="llama-3.3-70b-versatile"
    )
else:
    st.error("Missing API Key in Streamlit Secrets!")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    st.chat_message(role).write(msg.content)

if prompt := st.chat_input("Ask me anything..."):
    # Add user message
    new_user_msg = HumanMessage(content=prompt)
    st.session_state.messages.append(new_user_msg)
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = llm.invoke(st.session_state.messages)
                st.write(response.content)
                st.session_state.messages.append(AIMessage(content=response.content))
            except Exception as e:
                st.error(f"AI Error: {e}")
