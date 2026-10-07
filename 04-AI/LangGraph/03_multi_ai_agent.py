from dotenv import load_dotenv
load_dotenv()

import warnings
warnings.filterwarnings('ignore')

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing import Literal # 
from langchain.agents import create_agent
from langchain_community.utilities import GoogleSerperAPIWrapper # For Google Search Agent
from langchain.tools import tool


class FlowState(BaseModel):
    question:str = Field(description="User Asked Question")
    # Literal Limits to this Categories Only
    category: Literal['coding','google_search','weather'] = Field(default="google_search")  
    answer : str = Field(default="")

class QuestionCategory(BaseModel):
    # Literal Limits to this Categories Only
    category: Literal['coding','google_search','weather'] = Field(default="google_search", 
                                                                  description="Question Category")

# LLM Model
model = ChatGroq(
    model="openai/gpt-oss-120b"
)

# Agents =====
# Google-search Agent
search = GoogleSerperAPIWrapper()

@tool
def google_search(query: str):
    """
    Search Google and return relevant search results.
    """
    return search.run(query)

google_agent = create_agent(
    model=model,
    tools=[google_search],
    system_prompt="You are an agent that can search Google to answer questions."
)

# Weather Agent

@tool
def get_weather(city:str):
    """
    It provides real time Weather details for any city
    """
    return f"The current temperature in {city} is 33.C "
weather_agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="You are Agent and can provide realtime weather details "
)
#  ==========================

def check_question_category_temp(state:FlowState) -> FlowState:
    st_model = model.with_structured_output(QuestionCategory) #Output will be in QuestionCategory object
    res = st_model.invoke(f"""I want to Know the category of my question, Question is {state.question}.
                          If not sure then just give 'google_search' category""")
    state.category = res.category
    return state

def check_question_category(state: FlowState) -> FlowState:

    res = model.invoke(
        f"""
        Classify the following question into exactly ONE category.

        Categories:
        - coding
        - google_search
        - weather

        Question:
        {state.question}

        Return ONLY one of these exact values:
        coding
        google_search
        weather

        If the question does not clearly belong to coding or weather,
        return google_search.
        """
    )

    category = res.content.strip().lower()

    if category not in ["coding", "google_search", "weather"]:
        category = "google_search"

    state.category = category

    return state

def route(state:FlowState) -> Literal['coding','google_search','weather']:
    return state.category

def coding_node(state:FlowState) -> FlowState:
    res = model.invoke(f"You Are a Coding expert:{state.question}")
    state.answer = res.content
    return state

def weather_node(state:FlowState) -> FlowState:
    res = weather_agent.invoke(
        {
            "messages":[
                {
                    "role":"user",
                    "content":state.question
                }
            ]
        }
    )
    state.answer = res["messages"][-1].content
    return state

def google_search_node(state:FlowState) -> FlowState:
     res = google_agent.invoke(
            {
                "messages":[
                    {
                        "role":"user",
                        "content":state.question
                    }
                ]
            }
        )
     state.answer = res["messages"][-1].content
     return state

graph = StateGraph(FlowState)

# Category Node
graph.add_node("category",check_question_category)

# Coding node
graph.add_node("coding",coding_node)

# Google Search node
graph.add_node("google_search",google_search_node)

# Weather Node
graph.add_node("weather",weather_node)

# Edges
graph.add_edge(START,"category")    
# Conditional Edge
graph.add_conditional_edges("category",route)

graph.add_edge("coding",END)
graph.add_edge("google_search",END)
graph.add_edge("weather",END)

graph = graph.compile()

def save_graph_image():
    # Generate the graph image
    png_bytes = graph.get_graph().draw_mermaid_png()

    # Save it as a PNG file
    with open("Multi_agent_graph.png", "wb") as f:
        f.write(png_bytes)

    print("Graph saved successfully as Multi_agent_graph.png")

response = graph.invoke(
    {
        "question":"what is Address of VIT College Pune ?"
    }
)

from pprint import pprint

pprint(response)