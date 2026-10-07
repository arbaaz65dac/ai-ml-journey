from dotenv import load_dotenv
load_dotenv()

import warnings
warnings.filterwarnings('ignore')

from pydantic import BaseModel
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages  #Store User question and response in the form of list
from langgraph.checkpoint.memory import InMemorySaver
from typing import List, Annotated

# Sate
class ChatState(BaseModel):
    msg:Annotated[list, add_messages]

# LLM Model
model = ChatGroq(
    model="openai/gpt-oss-20b"
)

# Node Function
def chat_bot_node(state:ChatState) -> ChatState :
    response = model.invoke(state.msg)
    state.msg = [response]
    return state

# Memory 
memory = InMemorySaver()

# Graph
graph = StateGraph(ChatState)
graph.add_node("chatBot",chat_bot_node) # Node
graph.add_edge(START,"chatBot")  # Edge
graph.add_edge("chatBot",END)

# Compile Graph
final_graph = graph.compile(checkpointer=memory)

while True:
    query = input("User:")
    if query.lower() in ["quit","bye","end"]:
        print("Thanks for using me..!")
        break
    result = final_graph.invoke(
        {
            "msg":[
                {
                    "role":"user",
                    "content" : query
                }
            ]
        },
        {
            "configurable":{
                "thread_id" : "my-bot-1"
            }
        }
    )

    print("AI:",result["msg"][-1].content)