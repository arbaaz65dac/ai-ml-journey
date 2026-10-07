from pydantic import BaseModel # To maintain state
from langgraph.graph import StateGraph, START, END

class GreetState(BaseModel):
    msg:str = ""

graph = StateGraph(GreetState)  # Type of Graph Is GreetState


### Nodes Defining ->Py functions

def Greet(state:GreetState):
    state.msg = f"{state.msg}, Hello How's U ?"
    return state
def upper_case(state:GreetState):
    state.msg = state.msg.upper()
    return state

graph.add_node("greet",Greet) # Greet assigned as Node
# print(graph)  #<langgraph.graph.state.StateGraph object at 0x0000025A1CFCB8C0>
graph.add_node("upper",upper_case)
#              starts From,to which node
graph.add_edge(START, "greet") # Edge Assigned
graph.add_edge("greet","upper")  

graph.add_edge("upper",END)

final_graph = graph.compile()
response = final_graph.invoke({"msg":"I Love LangGraph !"})
print(response)  # {'msg': "I Love LangGraph !, Hello How's U ?"}

# Print Graph 
# from IPython.display import Image
# Image(final_graph.get_graph().draw_mermaid_png())

# Generate the graph image
png_bytes = final_graph.get_graph().draw_mermaid_png()

# Save it as a PNG file
with open("langgraph.png", "wb") as f:
    f.write(png_bytes)

print("Graph saved successfully as langgraph.png")
