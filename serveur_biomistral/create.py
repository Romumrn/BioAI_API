import requests

r = requests.post("http://localhost:8002/v1/api-keys",
                  params={"email": "test@example.com", "quota_tokens": 5000})

if r.ok:
    open("token.txt", "w").write(r.json()['api_key'])
    print("✅ Créé ! Token dans token.txt")
else:
    print("❌", r.text)
