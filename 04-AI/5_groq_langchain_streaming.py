from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    streaming=True
)
query = input("Ask Anything!:\n")
response = llm.stream(query)
print("response:\n",response) #<generator object BaseChatModel.stream at 0x000001F21061CDD0>
print("Type:\n",type(response)) # <class 'generator'>

# Now we have to iterate Generator 
for chunk in response :
    print(chunk.content, end="")
