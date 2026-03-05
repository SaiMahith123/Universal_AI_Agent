import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
import datetime

st.set_page_config(page_title="Universal AI Agent", layout="centered")
st.title("🤖 Universal AI Assistant")
if "GROQ_API_KEY" in st.secrets:
    llm = ChatGroq(
        groq_api_key=st.secrets["GROQ_API_KEY"], 
        model_name="mixtral-8x7b-32768",
        temperature=0
    )
else:
    st.error("Please add GROQ_API_KEY to Streamlit Secrets!")
    st.stop()
search_tool = DuckDuckGoSearchRun()
tools = [search_tool]
prompt_template = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt_template)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
if "messages" not in st.session_state:
    st.session_state.messages = []
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
if user_input := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)
with st.chat_message("assistant"):
    with st.spinner("Searching and thinking..."):
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        full_query = f"Today is {today}. {user_input}"
        try:
            response = agent_executor.invoke({"input": full_query})
            output = response["output"]
            st.write(output)
            st.session_state.messages.append({"role": "assistant", "content": output})
        except Exception as e:
            st.error(f"Error: {e}")
