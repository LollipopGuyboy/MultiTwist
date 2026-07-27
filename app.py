from flask import Flask, send_file, request, jsonify
from groq import Groq
import requests
import os

app = Flask(__name__)

# ----------------------------
# API KEYS
# ----------------------------

groq_key = os.environ.get("GROQ_API_KEY")
serper_key = os.environ.get("SERPER_API_KEY")

client = Groq(api_key=groq_key)

# ----------------------------
# WEB SEARCH
# ----------------------------

def search_web(query):
    url = "https://google.serper.dev/search"

    headers = {
        "X-API-KEY": serper_key,
        "Content-Type": "application/json"
    }

    response = requests.post(
        url,
        headers=headers,
        json={"q": query}
    )

    return response.json()

# ----------------------------
# HOME
# ----------------------------

@app.route("/")
def home():
    return send_file("multitwist.html")

# ----------------------------
# CHAT
# ----------------------------

@app.route("/chat", methods=["POST"])
def chat():

    data = request.json
    message = data.get("message", "")

    lower = message.lower()

    # Decide whether web search is needed
    search_words = [
        "latest",
        "today",
        "current",
        "news",
        "recent",
        "2025",
        "2026",
        "2027",
        "weather",
        "price",
        "stock",
        "live",
        "who won",
        "breaking",
        "update"
    ]

    need_search = any(word in lower for word in search_words)

    messages = [
        {
            "role": "system",
            "content": """
You are MultiTwist AI created by Rishabh.

Behave like ChatGPT.

- Be friendly.
- Talk naturally.
- If someone says hello, hi or hey, greet them.
- Never explain what a greeting means.
- Answer naturally.
- Keep responses clear and helpful.
- If web search information is provided, use it naturally.
"""
        }
    ]

    # Use web search only when needed
    if need_search:

        results = search_web(message)

        web_info = ""

        if "organic" in results:

            for item in results["organic"][:5]:

                web_info += f"Title: {item.get('title')}\n"
                web_info += f"Snippet: {item.get('snippet')}\n\n"

        messages.append({
            "role": "system",
            "content": f"Recent web information:\n\n{web_info}"
        })

    messages.append({
        "role": "user",
        "content": message
    })

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages
    )

    reply = response.choices[0].message.content

    return jsonify({
        "reply": reply
    })

# ----------------------------
# RUN
# ----------------------------

if __name__ == "__main__":
    app.run(debug=True)
