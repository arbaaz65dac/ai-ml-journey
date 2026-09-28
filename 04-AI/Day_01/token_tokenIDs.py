
import tiktoken

text = "Who is Rohit Sharma"

tokenizer = tiktoken.encoding_for_model(model_name='gpt-4')

tokenIDs = tokenizer.encode(text) 

print("Token IDs for text:",tokenIDs) # [15546, 374, 42087, 275, 61115]

result_tokenIds = [49, 2319, 275, 61115, 374, 264, 2294, 59019, 2851]

result = tokenizer.decode(result_tokenIds)

print("Result:", result)

