from dotenv import load_dotenv
load_dotenv()

import warnings
warnings.filterwarnings('ignore')

from langchain_community.document_loaders import PyPDFLoader  # For  Loading Document
from langchain_text_splitters import RecursiveCharacterTextSplitter # For Splittng/Chunks of Data
from langchain_google_genai import GoogleGenerativeAIEmbeddings # For Vector Embedding
from langchain_chroma import Chroma  # For Vector Store
from langchain_groq import ChatGroq # For LLM model
from langchain_core.prompts import PromptTemplate # For prompt Generation

##  Loading Data
loader = PyPDFLoader("data_science_syllabus.pdf")
docs = loader.load()
# print(len(docs))

##  Chunking Data
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,   # 1000  character in a chunk
    chunk_overlap = 200  # 200 character overlap b/w chunks
)

splitted_data = splitter.split_documents(docs)
# print(len(splitted_data))

## Vector Embedding 
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

## Vector Store In ChromaDB
vectore_store = Chroma.from_documents(
    documents=splitted_data,
    embedding=embeddings
)

## Similarity Search for User Query
query = "What topics are covered in Machine Learning and Data Science?"
data = vectore_store.similarity_search(query=query, k=2)  # By default top 4 similarity data give , k = 2 for top 2
# print(data[0].page_content)

## Context data for LLM
context = ""
for doc in data:
    context += doc.page_content + "\n" 
# print(context)

## LLM Model
model = ChatGroq(
    model="openai/gpt-oss-20b"
)

## Chain -> context_generate | prompt | llm | strparser

##  Context Generator
def get_context(query: str):

    data = vectore_store.similarity_search(
        query=query,
        k=4
    )

    # ========== RETRIEVED CONTEXT ==========
    context = ""
    for  doc in data:
        context += doc.page_content + "\n"
    return {
        "context": context,
        "question": query
    }

prompt = PromptTemplate.from_template(
    """You are a helpful Assistant.
    Provide the answer based only on the provided context.
    If you don't know the answer, say "I don't know."

    Context:
    {context}

    Question:
    {question}

    Answer:
"""
)

## RAG Chain
rag_chain = get_context | prompt | model 
# result = rag_chain.invoke("What is the duration of my course ?")
result = rag_chain.invoke("What are the contents of JAVA and DSA Module ?")
print(result.content)
