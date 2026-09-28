import requests

url = "https://jsonplaceholder.typicode.com/users"

response = requests.get(url)

print(response.status_code)

payload = {
    "name": "Arbaaz",
    "role": "AI Developer"
}

response1 = requests.post(url, json=payload)
print(response1.status_code)

data = response.json()

users = [user['name'] for user in data]
print(users)