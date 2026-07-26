from flask import Flask, send_file, request, jsonify
from groq import Groq
import requests
import os

app = Flask(__name__)

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY")
)


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

    web_info = ""

    try:
        search_results = search_web(message)

        if "organic" in search_results:
            for result in search_results["organic"][:5]:
                web_info += (
                    f"Title: {result.get('title', '')}\n"
                    f"Snippet: {result.get('snippet', '')}\n\n"
                )
    except Exception:
        web_info = ""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": """
You are MultiTwist AI created by Rishabh.

If live web search results are provided,
use them to answer with the latest information.

If no web results are available,
answer normally.
"""
            },
            {
                "role": "user",
                "content": f"""
User Question:

{message}

Live Web Search Results:

{web_info}
"""
            }
        ]
    )

    reply = response.choices[0].message.content

    return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run(debug=True)
