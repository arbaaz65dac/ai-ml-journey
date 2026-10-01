from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq #model
from langchain_community.utilities import GoogleSerperAPIWrapper #tool
from langchain.agents import create_agent  # agent
from langgraph.checkpoint.memory import MemorySaver # Short term memory
import streamlit as st

llm = ChatGroq(
    model="openai/gpt-oss-20b"
)
search = GoogleSerperAPIWrapper()
memory = MemorySaver()

agent = create_agent(
    model=llm,
    tools=[search.run],
    checkpointer=memory, # Impl Short Memory 
    system_prompt="You are a agent and can search any question on google."
)

while True:
    query = input("User: ")
    if query.lower() == "quit":
        print("Good Byeeee")
        break
    response = agent.invoke(
        {"messages":[{"role":"user", "content":query}]},
        {"configurable":{"thread_id": "asd123"}}
    )
    print("AI: ",response["messages"][-1].content)
