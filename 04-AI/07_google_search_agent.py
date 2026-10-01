from dotenv import load_dotenv
load_dotenv()


from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_groq import ChatGroq
from langchain.agents import create_agent

llm = ChatGroq(
    model="openai/gpt-oss-20b"
)
search = GoogleSerperAPIWrapper()
# search.run("") run tool 

agent = create_agent(
    model=llm,
    tools=[search.run],
    system_prompt="You are a agent and can search any question on google."
)

query = input("Ask your Question\n")
response = agent.invoke({"messages":[{"role":"user", "content":query}]})

print("Result:\n",response["messages"][-1].content)

