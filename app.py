import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
import datetime

st.set_page_config(page_title="Universal AI Agent", layout="centered")
st.title("🤖 Universal AI Agent")

search_tool = DuckDuckGoSearchRun()

if "GROQ_API_KEY" in st.secrets:
    llm = ChatGroq(
        groq_api_key=st.secrets["GROQ_API_KEY"], 
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )
else:
    st.error("Please add GROQ_API_KEY to Streamlit Secrets!")
    st.stop()

agent = initialize_agent(
    tools=[search_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching the web and thinking..."):
            today = datetime.datetime.now().strftime("%Y-%m-%d")
            full_query = f"Today is {today}. {prompt}"
            try:
                response = agent.run(full_query)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error: {e}")

with st.sidebar:
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
