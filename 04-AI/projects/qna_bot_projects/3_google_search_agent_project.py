from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq #model
from langchain_community.utilities import GoogleSerperAPIWrapper #tool
from langchain.agents import create_agent  # agent
from langgraph.checkpoint.memory import MemorySaver # Short term memory
import streamlit as st

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    streaming=True
)
search = GoogleSerperAPIWrapper()

if "memory" not in st.session_state:  
    st.session_state.memory = MemorySaver()  ## Memory Store
    st.session_state.history = []   ## History Store

agent = create_agent(
    model=llm,
    tools=[search.run],
    checkpointer=st.session_state.memory, # Impl Short Memory 
    system_prompt="You are a agent and can search any question on google."
)

### Building Web interface
st.subheader("Quick Answer : answer at speed of thought")

for msg in st.session_state.history: ## Printing History On Ui
    role = msg['role']
    content = msg['content']
    st.chat_message(role).markdown(content)
 

query = st.chat_input("Ask Anything!")
if query:
    st.chat_message("user").markdown(query)
    st.session_state.history.append({"role":"user","content":query})
    response = agent.stream(
    {"messages":[{"role":"user", "content":query}]},
    {"configurable":{"thread_id": "asd123"}},
    stream_mode="messages"
    )
    ai_container = st.chat_message("ai")
    with ai_container:
        space = st.empty()
        msg = ""  
        for chunk in response:
            msg = msg + chunk[0].content
            space.write(msg)
    
        st.session_state.history.append({"role":"ai","content":msg})
    

    