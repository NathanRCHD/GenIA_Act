import requests

token = "sk-dac51a8d68f449ff9e8f7224fb43e149"

def chat_with_model(token, message="bonjour"):
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
          "content": message
        }
      ]
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()

#load csv and iterate over rows to chat with model
import pandas as pd
df = pd.read_csv('eu_ai_act_qna_gold.csv')
for index, row in df.iterrows():
    print(f"Row {index}: {row['question']}")
    response = chat_with_model(token, message=row['question'])
    print(f"Response: {response}")