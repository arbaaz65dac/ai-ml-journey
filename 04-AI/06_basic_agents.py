from dotenv import load_dotenv
load_dotenv()

from langchain.tools import tool   # For @tool decorator
#  For tools 2 things are important @tool decorator and doc string """ """.
@tool
def add_numbers(a:int , b:int):
    #  Defining the doc String (Explain What our function is going to do)
    """" 
        it will return the sum of two numbers
        Args :
            a : Number one
            b : Number two 
    """
    return a+b

@tool
def mult_numbers(a:int , b:int):
    #  Defining the doc String (Explain What our function is going to do)
    """" 
        it will return the Multiplaction of two numbers
        Args :
            a : Number one
            b : Number two 
    """
    return a * b

result = add_numbers.invoke({"a":12 , "b":44})
# print(result)

from langchain_groq import ChatGroq
from langchain.agents import create_agent # used to link tool and llm 

llm = ChatGroq(model="openai/gpt-oss-20b")

agent = create_agent(
    model=llm,
    tools = [add_numbers, mult_numbers],
    system_prompt="You are a math teacher and always tools for calculation"  #tells system how to act
)

# Agent invoking

response = agent.invoke({"messages":[{"role":"user", "content":"What is 2 + 3 * 25"}]})

for res in response["messages"]:
    print(res)
    print("\n")

# content='What is 2 + 3 * 25' 
# additional_kwargs={} response_metadata={} id='ccb8b0ee-5bdb-4ffd-9bd0-4ca6ea424694'


# content='' 
# additional_kwargs={'reasoning_content': 'User asks: "What is 2 + 3 * 25". 
# We need to compute. According to order of operations, 
# multiplication first: 3*25=75. Then 2+75=77. So answer 77. 
# Use tool? We can use add_numbers for 2 and 75. But we can also compute directly.
# The instructions: "You are a math teacher and always tools for calculation". 
# So we should use the tool to calculate. We can compute 3*25 using mult_numbers, 
# then add 2. Let\'s do that.', 'tool_calls': [{'id': 'fc_de5a249c-7195-4bf4-af60-79a7b56dd8f7', 
# 'function': {'arguments': '{"a":3,"b":25}', 'name': 'mult_numbers'}, 'type': 'function'}]}
# response_metadata={'token_usage': {'completion_tokens': 143, 'prompt_tokens': 220, 'total_tokens': 363, 
# 'completion_time': 0.201860729, 'completion_tokens_details': {'reasoning_tokens': 116}, 
# 'prompt_time': 0.012530256, 'prompt_tokens_details': None, 'queue_time': 0.047119353, 'total_time': 0.214390985}, 
# 'model_name': 'openai/gpt-oss-20b', 'system_fingerprint': 'fp_e99e93f2ac', 'service_tier': 'on_demand', 
# 'finish_reason': 'tool_calls', 'logprobs': None, 'model_provider': 'groq'} id='lc_run--01a0f6fd-fcb2-7b50-8836-92958c4f29d3-0' 
# tool_calls=[{'name': 'mult_numbers', 'args': {'a': 3, 'b': 25}, 'id': 'fc_de5a249c-7195-4bf4-af60-79a7b56dd8f7', 'type': 'tool_call'}] invalid_tool_calls=[] usage_metadata={'input_tokens': 220, 'output_tokens': 143, 'total_tokens': 363, 'output_token_details': {'reasoning': 116}}


# content='75' name='mult_numbers' id='37c66c73-8bad-4dd2-a185-0ec02d1cd3a7' tool_call_id='fc_de5a249c-7195-4bf4-af60-79a7b56dd8f7'


# content='The result is **77**.' 
# additional_kwargs={'reasoning_content': 'User: "What is 2 + 3 * 25". We need to compute. The order: multiplication first: 3*25=75, then 2+75=77. 
# The user likely wants the result. We should respond with 77.'} 
# response_metadata={'token_usage': {'completion_tokens': 71, 'prompt_tokens': 253, 'total_tokens': 324, 'completion_time': 0.075091288, 'completion_tokens_details': {'reasoning_tokens': 55}, 'prompt_time': 0.015343, 'prompt_tokens_details': None, 'queue_time': 0.333697514, 'total_time': 0.090434288}, 'model_name': 'openai/gpt-oss-20b', 'system_fingerprint': 'fp_851252edc2', 'service_tier': 'on_demand', 'finish_reason': 'stop', 'logprobs': None, 'model_provider': 'groq'} id='lc_run--01a0f6fd-fe9b-7403-8bd0-bd004556ff8a-0' tool_calls=[] invalid_tool_calls=[] usage_metadata={'input_tokens': 253, 'output_tokens': 71, 'total_tokens': 324, 'output_token_details': {'reasoning': 55}}

print("Result:\n",response["messages"][-1].content) #  The result is **77**.