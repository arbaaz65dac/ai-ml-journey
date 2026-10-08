from dotenv import load_dotenv
load_dotenv()
import warnings
warnings.filterwarnings('ignore')

from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from langgraph.types import interrupt  #To stop at particular node for user feedback and questioning
from langgraph.types import Command #To give command to LLM about feed Back
from langgraph.checkpoint.memory import InMemorySaver
from typing import Literal

# State
class EmailState(BaseModel):
    query: str = ""
    draft : str = ""
    human_feedback : str = ""
    final_response : str = ""

# LLM model
model = ChatGroq(
    model="openai/gpt-oss-20b"
)

# Draft email, Human feedback, Finalize (NODES)
# Draft Email Node
def draft_email(state:EmailState) -> EmailState:

    if state.human_feedback:
        draft = model.invoke(
            f"""
                Rewrite this draft email for :{state.query},
                You have generated this mail:{state.draft}, 
                human feedback to applied : {state.human_feedback}
            """
        ).content
    else:
        draft = model.invoke(state.query).content
    state.draft = draft
    return state

# Human feedback Node
def human_feedback(state:EmailState) -> EmailState:
    feedback = interrupt(
        {
            "draft_email":state.draft,
            "question":"Do you want to continue or rewrite mail"
        }
    )

    fb = (feedback or "").strip().lower()
    if fb in ("approved","ok","yes"):
        state.human_feedback = ""
        return state
    else:
        state.human_feedback = fb
        return state

# Finalize Node
def finalize_node(state:EmailState) -> EmailState:
    if state.human_feedback :
        state.final_response = state.human_feedback
        return state
    else:
        print("Email sent successfully")
        state.final_response = "Email sent successfully"
        return state
    
# Conditional Routing
def conditional_routing(state:EmailState) -> Literal["draft_email","finalize_node"]:
    if state.human_feedback:
        return "draft_email"
    return "finalize_node"


# Graph 
graph_builder = StateGraph(EmailState)

# Nodes
graph_builder.add_node("draft_email",draft_email)
graph_builder.add_node("human_feedback",human_feedback)
graph_builder.add_node("finalize_node",finalize_node)

# Edges
graph_builder.add_edge(START,"draft_email")
graph_builder.add_edge("draft_email","human_feedback")
graph_builder.add_conditional_edges("human_feedback",conditional_routing)
graph_builder.add_edge("finalize_node",END)

graph = graph_builder.compile(
    checkpointer= InMemorySaver()
)

def save_graph_image():
    # Generate the graph image
    png_bytes = graph.get_graph().draw_mermaid_png()

    # Save it as a PNG file
    with open("Human_in_the_loop_graph.png", "wb") as f:
        f.write(png_bytes)

    print("Graph saved successfully as Human_in_the_loop_graph.png")

config = {
    "configurable":{
        "thread_id":1
    }
}

result = graph.invoke(
    {
        "query": "I want to Send a email for Medical Leave request, dates will be 11 oct to 15 oct , so create a simple mail"
    },config
)
print(result["draft"])

result = graph.invoke(
    Command(
        resume= "I want to change the topic for leave not medical leave it should be Sick Leave"
        # resume="yes"
    ),
    config=config
)

print("+++++++++++++result+++++++++++++")
print(result["draft"])
print("+++++++++++++result+++++++++++++")
result = graph.invoke(
    Command(
        resume="yes"
    ),
    config=config
)
print(result["final_response"])