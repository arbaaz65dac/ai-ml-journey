from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("API_KEY")

if api_key :
    print("Successfully Loaded")
else :
    print("Error In Loading Key")