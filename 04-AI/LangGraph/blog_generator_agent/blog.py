from dotenv import load_dotenv
load_dotenv()
import warnings 
warnings.filterwarnings('ignore')
from state import BlogState
from agents import get_llm, researcher_agent

llm = get_llm()
outline = researcher_agent(llm=llm, topic="What is LLM", audience="Developers")
# print(outline)

from graph import build_blog_graph
from langgraph.types import Command

graph = build_blog_graph()

config = {
    "configurable":{
        "thread_id":1
    }
}

state = graph.invoke(
    {
        "topic":"What Is MCP (Model Context Protocal) ? A Developer's Guide for 2026",
        "audience":"ai and agentic ai developers"
    }, 
    config = config
)

snap = graph.get_state(config)
interrupt_payload = snap.interrupts[0].value
# print("Stage:",interrupt_payload.get("stage"))
# print("\n------- Research Outline for your review -------\n")
# print(interrupt_payload.get("research"))

state = graph.invoke(
    Command(
        resume={"action":"approve","feedback":""}
    ),
    config=config
)

snap = graph.get_state(config)
interrupt_payload = snap.interrupts[0].value
# print("Stage:",interrupt_payload.get("stage"))
# print("\n------- Draft Blog for your review -------\n")
# print(interrupt_payload.get("draft"))

# Approve the draft Blog
state = graph.invoke(
    Command(
        resume={"action":"approve","feedback":""}
    ),
    config=config
)

# print("----- Final Published Blog ------")
# print(state["final_blog"])

# Review Flow / Human Feedback flow

# graph.invoke(Command(
#         resume={
#             "action":"revise","feedback":"Rewrite the blog title and more focus on codes"
#         }
#     ),config=config
#     )