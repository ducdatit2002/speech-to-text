import requests

url = "http://0.0.0.0:8000/summarize"  # Endpoint /summarize đã thêm trong app.py

with open('demo15.txt', 'r', encoding='utf-8') as f:
    text_content = f.read().strip()
 
data = {
    "text": text_content
}

response = requests.post(url, json=data)

if response.status_code == 200:
    print("Tóm tắt:", response.json()['summary'])
else:
    print("Lỗi:", response.text)
