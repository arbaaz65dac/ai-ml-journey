from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()
key = os.getenv("GOOGLE_API_KEY")


llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    api_key=key
)


# Standard protocal to pass prompts if list of dictionary
prompts = [
    {"role":"system", "content":"You are a JS developer"},
    {"role":"user", "content":"Diff b/w map and filter"}
]

# response = llm.invoke(prompts)

# print(response.content)

# PROMPTS TEMPLETES CHAINING (Dynamin Prompts)
from langchain_core.prompts import ChatPromptTemplate

prompts_tempeletes = ChatPromptTemplate.from_messages([
    #                                     Dynamic Prompting 
    {"role":"system", "content":"You are a {language} developer"},
    {"role":"user", "content":"{query}"}
])
# print(prompts_tempeletes)

final_prompt = prompts_tempeletes.format_messages(language="Python", query="How to sort an array?")
# print(final_prompt)

# response = llm.invoke(final_prompt)
# print(response.content)

# Chains :
#  ->Runnables - the function/data we can invoke is runnable
prompts_tempeletes.invoke({"language":"python", "query": "How to sort Array"})
# llm.invoke() ==> also runnable
# llm.invoke(prompts_tempeletes.invoke({"language":"python", "query": "How to sort Array"})) chaining


prompts_tempeletes_1 = ChatPromptTemplate.from_messages([
    #                                     Dynamic Prompting 
    {"role":"system", "content":"You are a Translator and translet input in {language}"},
    {"role":"user", "content":"{query}"}
])
from langchain_core.output_parsers import StrOutputParser
output = StrOutputParser()

# ========== MY SELF CHAIN NODE ===========
def transform_case(result:str):
    return result.upper()

chains = prompts_tempeletes_1 | llm | output | transform_case
response = chains.invoke({"language":"hinglish", "query": "I Love LangChain !"})
# print(response.content) now no need to do .content bcoz we have used outputparser
print(response)





