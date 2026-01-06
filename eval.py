import requests

token = "sk-dac51a8d68f449ff9e8f7224fb43e149"

def chat_with_model(token):
    url = 'https://k2vm-74.mde.epf.fr/api/chat/completions'
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    data = {
      "model": "n8n",
      "messages": [
        {
          "role": "user",
          "content": "bonjour"
        }
      ]
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()

print(chat_with_model(token))