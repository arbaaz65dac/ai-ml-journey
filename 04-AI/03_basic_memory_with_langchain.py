from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash"
)

# Convesritiotn History Providing
prompts = [
    {"role":"user", "content":"Hello, My name is Arbaaz Shaikh"},
    {"role":"ai", "content":"Hello, Arbaaz Shaikh! How can I assist you today?"},
    {"role":"user", "content":"What is my name?"}
]

response = llm.invoke(prompts)
print(response.content) 

# === Manage History =====

history = []

while True :
    query = input("User:")
    if query.lower() in ['exit','bye','quit']:
        print("Good Bye!!")
        break

    history.append({"role":"user", "content":query})
    print("User:",query)


    response_1 = llm.invoke(history)
    history.append({"role":"ai", "content":response_1.content})

    print("AI:",response_1.content, "\n")