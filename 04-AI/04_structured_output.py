from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash"
)
text = "Hello My Name is arbaaz Shaikh, my email is arbaazshaikh@gmail.com and I'm 24 years old"

from pydantic import BaseModel, Field

class ResponseStructure(BaseModel):
    name:str = Field(description="Complete Name")
    email:str = Field(description="E-mail Address")
    age:int = Field(description="Age")

structured_llm = llm.with_structured_output(ResponseStructure)  #Linking with structured output

response = structured_llm.invoke("Please give me only name, email and age from this text:{text}")
print(response.model_dump())

class Movies(BaseModel):
    title:str = Field(description="Movie Title")
    year:int = Field(description="Movie released year")

from typing import List

class AllMovies(BaseModel):
    movies:list[Movies]


movie_llm = llm.with_structured_output(Movies)

# For Single
response_movie = movie_llm.invoke("Give me one Sci-Fi movie")
print(response_movie)  #  Movies(title="The Matrix", year = 1999)


# For Multiple
all_movie_llm = llm.with_structured_output(AllMovies)
response_movie = movie_llm.invoke("Give me three Sci-Fi movie")
print(response_movie) 

