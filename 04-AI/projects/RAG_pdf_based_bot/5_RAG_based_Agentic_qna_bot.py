from dotenv import load_dotenv
load_dotenv()

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader ## For multiple files (directory loads)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_groq.chat_models import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

### Data in st session 
if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "agent" not in st.session_state:
    st.session_state.agent = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []


def process_document(path):

    # Document Loading
    loader = PyPDFDirectoryLoader(path)
    docs = loader.load()

    # Splitting into Chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200
    )
    splitted_docs = splitter.split_documents(docs)

    # Embeddings and vectoreDB
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )
    vector_db = Chroma.from_documents(
        documents=splitted_docs,
        embedding=embeddings
    )

    ## Agent Creating:

    # LLM Model 
    model = ChatGroq(
        model="openai/gpt-oss-20b"
    )

    # Tool
    @tool
    def retrieve_context(query:str):
        """
        Retrieve documents relevent to a query from the knowledge base
        """
        context = ""
        data = vector_db.similarity_search(
            query=query,
            k=4
        )
        for doc in data:
            context += doc.page_content + "\n"
        return context

    # System Prompt
    system_prompt = """
        You are a helpful assistant that answers questons using retrieved context.
        My knowledge base consists of the details from the uploaded document.
        Always use the 'retriever_context' tool for questions requring external knowledge
    """
    # Memory
    memory =  InMemorySaver()

    # Agent
    agent = create_agent(
        model=model,
        tools=[retrieve_context],
        system_prompt=system_prompt,
        checkpointer=memory
    )

    st.session_state.agent = agent
    st.session_state.document_uploaded = True

### Upload UI

if not st.session_state.document_uploaded:
    st.header("📂 Upload Documents")
    uploaded = st.file_uploader(label="Select PDF Files",type=["pdf"], accept_multiple_files=True)
    if uploaded:
        with st.spinner("Analyzing..."):
            path = "./docs_files/"
            for file in uploaded:
                with open(path + file.name,"wb") as f:
                    f.write(file.getvalue())
            process_document(path)
            st.rerun() ## Variables/UI are reloaded based on updated data/session variables

### Chat UI

if st.session_state.document_uploaded and st.session_state.agent:
    st.subheader("🤖 How can I help you with your Uploaded Documents?")
    for msg in st.session_state.messages:
        role = msg.get("role")
        content = msg.get("content")
        st.chat_message(role).markdown(content)
    query = st.chat_input("Ask questions about your uploaded documents...")
    if query :
        st.session_state.messages.append(
            {
                "role":"user",
                "content":query
            }
        )
        st.chat_message("user").markdown(query)
        response = st.session_state.agent.invoke(
            {
                    "messages":[
                        {
                            "role":"user",
                            "content":query
                        }
                    ]
            },
            {
                "configurable":{"thread_id":1}
            }
        )
        answer = response["messages"][-1].content
        st.chat_message("ai").markdown(answer)
        st.session_state.messages.append(
            {
                "role":"ai",
                "content":answer
            }
        )



