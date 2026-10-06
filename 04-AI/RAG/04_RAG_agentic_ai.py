from dotenv import load_dotenv
load_dotenv()

import warnings
warnings.filterwarnings('ignore')

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import InMemoryVectorStore  # Using Inmemory DB instead of ChromaDB
from langchain.tools import tool
from langchain.agents import create_agent

loader = PyPDFLoader("medical_report.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

splitted_docs = splitter.split_documents(documents=docs)

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

vector_store = InMemoryVectorStore.from_documents(
    documents=splitted_docs,
    embedding=embeddings
)

## Building AI Agent -> calls a tool according to user query and fetch most similary record from Vector db
# Agent -> tools, LLM, Prompt

@tool
def retriever_tool(query:str):
    """
    Search the medical report for relevant information.

    Use this tool whenever the user asks about information contained
    in the uploaded medical report, such as patient details, doctor
    names, diagnosis, medications, test results, dates, or other
    report information.
    """
    print("Tool Called",query)
    data = vector_store.similarity_search(query=query, k=4)
    context = ""
    for doc in data:
        context += doc.page_content + "\n"
    return context

model = ChatGroq(
    model="openai/gpt-oss-20b"
)

system_prompt = """
    You are a helpful assistant that answers questons using retrieved tool context.
    always use the 'retriever_tool' tool for questions requring external knowledge
"""
agent = create_agent(
    model=model,
    tools=[retriever_tool],
    system_prompt=system_prompt
)

query = "What is the name of patient, and what is name of doctors ? "
response = agent.invoke(
    {
        "messages":[
            {
                "role":"user",
                "content":query
            }
        ]
    }
)

result = response["messages"][-1].content
print(result) 

