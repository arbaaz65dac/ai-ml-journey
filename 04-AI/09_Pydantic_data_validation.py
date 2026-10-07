class User:
    name = ""
    email = ""
    age = 0

user = User()
user.name = "Rahul"
user.email = "rahul@gmail.com"
user.age = "My age is 23"  # Should Give error but 
# print(user)  # <__main__.User object at 0x000001E2FB656BA0>  Still works correctly and give Object of user
# print(user.age) # My age is 23
# TO overcome this issues we use Pydantic Library
from pydantic import BaseModel, ConfigDict, Field
class Student(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    name : str = ""   # Pydantic provide def to data type
    age : int = 0

student = Student()
student.name = "Raj"
student.age = "23" # Pydantic converts: "23":str -> 23:int bcoz "23" is a valid integer representation.
# print(student)

# Field Validation
class Product(BaseModel):
    title: str = Field(max_length=25, min_length=2) # Max Length 25 char min 2 char
    description : str = Field(min_length=10) #  Min length 10 char
    price : int = Field(gt=0, lt=150)  # Greater than 0 and Less Than 150

# prodt = Product(title="A",description="This is a excellent Mobile",price=23)
# print(prodt) String should have at least 2 characters [type=string_too_short, input_value='A', input_type=str]
prodt = Product(title="A10pro",description="This is a excellent Mobile",price=23)
# print(prodt)

from typing import List  # For Cart there may be multiple items of product so we have to pass it as List
from typing import Annotated # Gives Permission to Create custom/comman Validater
# Annotated
commanValidator =   Annotated[List[Product], "This is Comman Validator"]
class Cart(BaseModel):
    id:str 
    # item : List[Product]
    item : commanValidator
    comment : str = Field(default="")  # Default  

