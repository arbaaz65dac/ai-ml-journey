# LangChain Model - Google genAI
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()
key = os.getenv("GOOGLE_API_KEY")


llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    api_key=key
)

# PROMPTS (STATIC PROMPTS)
prompts = [
    # TYPE "System" -> Helps LLMs to understand how to perform
    ("system","You are a python developer"),
    # TYPE "User" -> Tells LLM It's users question/query
    ("user","How to sort an Array"),
    #Type "AI" -> Tells LLM It's your answer only
    

]
# response = llm.invoke("Who Is Rohit Sharma")
response = llm.invoke(prompts)
print(response.content)  
