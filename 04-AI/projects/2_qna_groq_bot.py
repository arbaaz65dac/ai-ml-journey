from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_groq import ChatGroq

# Groq LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    streaming=True
)

# Page
st.title("🤖 AskBuddy - AI QnA Bot")
st.markdown("My QnA Bot with LangChain and Groq!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]

    st.chat_message(role).markdown(content)

# Chat input
query = st.chat_input("Ask Anything!")

if query:

    # Display user message
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    st.chat_message("user").markdown(query)

    # Create one assistant message container
    with st.chat_message("assistant"):

        message_placeholder = st.empty()

        full_response = ""

        # Stream response
        for chunk in llm.stream(query):

            full_response += chunk.content

            message_placeholder.markdown(full_response)

    # Save complete response
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response
    })