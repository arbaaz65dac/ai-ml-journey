from dotenv import load_dotenv
load_dotenv()
import warnings
warnings.filterwarnings('ignore')

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

# Document Load
loader = PyPDFLoader("Report.pdf")
docs = loader.load()

# Split/chunk
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)
docs = splitter.split_documents(docs)

# Embeddings
embedding = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

# Vector Store
vector_store = InMemoryVectorStore.from_documents(
    documents = docs,
    embedding=embedding
)

# LLM model
model = ChatGroq(
    model="openai/gpt-oss-20b"
)

# State
class RagState(BaseModel):
    question: str = Field(description="User Question")
    documents : list = []
    context : str = Field(description="Context data for user question",default="")
    answer : str = Field(description="Final Answer",default="")

#  Question -> Retrieve -> Context -> Generate -> End  (Nodes)

# Retrieve Node
def retrieve_node(state:RagState) -> RagState:
    docs = vector_store.similarity_search(
        query=state.question
    )
    state.documents = docs
    return state

# Context Node
def create_context_node(state:RagState) -> RagState:
    context = ""
    for doc in state.documents:
        context += doc.page_content + "\n\n"
    state.context = context
    return state

# Generate Answer Node
def generate_node(state:RagState) -> RagState:
    prompt = f"""You are a helpful assistatnt and provide the answer for user question based on provided context,
    If you don't find relevent answer just say I dont know.
    context is: {state.context},
    question is:{state.question}"""

    res = model.invoke(
        prompt
    )
    state.answer = res.content
    return state

# Graph 
graph = StateGraph(
    state_schema=RagState
)

# Nodes
graph.add_node("retrieve_node",retrieve_node)
graph.add_node("create_context_node",create_context_node)
graph.add_node("generate_node",generate_node)

# Edges
graph.add_edge(START,"retrieve_node")
graph.add_edge("retrieve_node","create_context_node")
graph.add_edge("create_context_node","generate_node")
graph.add_edge("generate_node",END)

graph = graph.compile()

def save_graph_image():
    # Generate the graph image
    png_bytes = graph.get_graph().draw_mermaid_png()

    # Save it as a PNG file
    with open("RAG_Based_graph.png", "wb") as f:
        f.write(png_bytes)

    print("Graph saved successfully as RAG_Based_graph.png")

res = graph.invoke(
    {
        "question": "What is the Topic and Give Short Summary of the report"
    }
)
answer = res["answer"]
print(answer)