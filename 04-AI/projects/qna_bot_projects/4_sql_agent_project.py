from dotenv import load_dotenv 
load_dotenv() 
 
from langchain_groq import ChatGroq 
from langchain_community.utilities import SQLDatabase 
from langchain_community.agent_toolkits import SQLDatabaseToolkit 
from langgraph.checkpoint.memory import InMemorySaver 
import streamlit as st 
 
db = SQLDatabase.from_uri("sqlite:///my_tasks.db") 
db.run(""" 
    CREATE TABLE IF NOT EXISTS tasks( 
        id INTEGER PRIMARY KEY AUTOINCREMENT, 
        title TEXT NOT NULL, 
        description TEXT, 
        status TEXT CHECK(status IN ('PENDING', 'IN_PROGRESS','COMPLETED')) DEFAULT 'PENDING', 
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP 
    );     
""") 
 
model = ChatGroq( 
    model="openai/gpt-oss-20b" 
) 
 
toolkit = SQLDatabaseToolkit(db=db,llm=model) 
tools = toolkit.get_tools() 
system_prompt = """ 
You are a task management assistant that interacts with a SQLite database
containing a 'tasks' table.

Your job is to help users create, read, update, and delete tasks.

IMPORTANT SQL RULES:

1. Use the provided SQL database tools to perform database operations.
2. Never invent table names or column names.
3. The only table you should modify is the 'tasks' table.
4. SELECT queries must return a maximum of 10 rows.
5. When listing tasks, use:
   ORDER BY created_at DESC
   LIMIT 10
6. After INSERT, UPDATE, or DELETE, execute a SELECT query to verify
   that the operation was successful.
7. Use only these status values:
   PENDING
   IN_PROGRESS
   COMPLETED
8. Do not put explanations, markdown, or reasoning inside SQL queries.
9. SQL tool arguments must contain ONLY valid JSON.
10. When the user asks to see tasks, present the result as a clean
    Markdown table.

TABLE SCHEMA:

tasks(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT,
    status TEXT,
    created_at TIMESTAMP
)

CRUD OPERATIONS:

CREATE:
INSERT INTO tasks(title, description, status)

READ:
SELECT * FROM tasks
ORDER BY created_at DESC
LIMIT 10

UPDATE:
UPDATE tasks
SET status = ?
WHERE id = ? OR title = ?

DELETE:
DELETE FROM tasks
WHERE id = ? OR title = ?

Always use the SQL database tools available to you. 
""" 
from langchain.agents import create_agent 
 
@st.cache_resource   # Dont recall this so memory refresh protected 
def get_agent(): 
    agent = create_agent( 
        model=model, 
        tools=tools, 
        checkpointer=InMemorySaver(), 
        system_prompt=system_prompt 
    ) 
    return agent 
 
agent = get_agent() 
 
st.subheader("📋TaskBot - Manage your Tasks") 
st.caption("Powered by Groq and MySQL",text_alignment="justify") 
 
if "messages" not in st.session_state: 
    st.session_state.messages = [] 
 
for msg in st.session_state.messages: 
    st.chat_message(msg["role"]).markdown(msg["content"]) 
 
prompt = st.chat_input("Ask me to Manage Your tasks") 
if prompt: 
    st.chat_message("user").markdown(prompt) 
    st.session_state.messages.append({"role":"user", "content":prompt}) 
    with st.chat_message("ai"): 
        with st.spinner("Processing..."): 
            response = agent.invoke( 
                {"messages":[{"role":"user","content":prompt}]}, 
                {"configurable":{"thread_id":"1"}}             
            ) 
 
            result = response["messages"][-1].content 
            st.markdown(result) 
            st.session_state.messages.append({"role":"ai", "content":result}) 
 