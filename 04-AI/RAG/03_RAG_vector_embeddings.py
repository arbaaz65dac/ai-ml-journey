from dotenv import load_dotenv
load_dotenv()

documents = [
    "Java is used for backend development.",
    "Spring Boot simplifies Java application development.",
    "React is used to build web interfaces.",
    "MongoDB is a NoSQL database.",
    "MySQL is a relational database.",
    "Python is widely used in machine learning.",
    "Neural networks are inspired by biological neurons.",
    "Transformers are important architectures for modern LLMs.",
    "Docker packages applications into containers.",
    "Git is a version control system."
]

from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

vector = embeddings.embed_documents(documents)
# print(len(vector[0]))

# Vector Store Chroma DB
from langchain_chroma import Chroma
vector_store = Chroma.from_texts(
    texts=documents,
    embedding= embeddings,
    persist_directory="vector_db"     # For to store the embedded data permentally into local store 
)

query = "Neural networks"
result = vector_store.similarity_search(query=query, k=2)
print(result[0].page_content)