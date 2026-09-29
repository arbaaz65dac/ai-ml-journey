from dotenv import load_dotenv
load_dotenv()
import streamlit as st 

from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(
    model = "gemini-3.8-flash"
)

st.title("🤖 AskBuddy - AI QnA Bot")
st.markdown("My QnA Bot with Langchain and google Gemini !")
if "messages" not in st.session_state: 
    st.session_state.messages = []

for message in st.session_state.messages :
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)

query = st.chat_input("Ask Anything !")

if query:
   st.session_state.messages.append({"role":"user", "content":query})
   st.chat_message("user").markdown(query)
   response = llm.invoke(query)
   st.chat_message("ai").markdown(response.content)
   st.session_state.messages.append({"role":"ai", "content":response.content})

