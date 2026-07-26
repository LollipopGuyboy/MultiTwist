import requests
from flask import Flask, send_file, request, jsonify
from groq import Groq
import os

app = Flask(__name__)

groq_key = os.environ.get("GROQ_API_KEY")

print("GROQ KEY EXISTS:", groq_key is not None)
print("GROQ KEY STARTS WITH:", groq_key[:4] if groq_key else "NONE")

client = Groq(api_key=groq_key)

def search_web(query):
    url = "https://google.serper.dev/search"

    headers = {
        "X-API-KEY": os.environ.get("SERPER_API_KEY"),
        "Content-Type": "application/json"
    }

    response = requests.post(
        url,
        headers=headers,
        json={"q": query}
    )

    return response.json()
@app.route("/")
def home():
    return send_file("multitwist.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    message = data.get("message", "")

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You are MultiTwist AI. You were created by Rishabh."
            },
            {
                "role": "user",
                "content": message
            }
        ]
    )

    reply = response.choices[0].message.content

    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(debug=True)
