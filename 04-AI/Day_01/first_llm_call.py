from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

respone = client.responses.create(
    model="gpt-4o-mini",
    input="Who is Rohit Gurunath Sharma"
)

print(respone.output_text)
