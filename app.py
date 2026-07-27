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

    # Decide if web search is needed
    search_words = [
        "latest", "today", "current", "news", "recent",
        "2025", "2026", "2027",
        "weather", "price", "stock", "live",
        "who won", "breaking", "update"
    ]

    need_search = any(word in message.lower() for word in search_words)

    messages = [
        {
            "role": "system",
            "content": """You are MultiTwist AI created by Rishabh.

Be friendly and conversational.
If someone says 'hi', 'hello', or 'hey', greet them naturally.
Do not explain greetings.
Answer naturally like ChatGPT.
"""
        }
    ]

    if need_search:
        results = search_web(message)

        web_info = ""

        if "organic" in results:
            for item in results["organic"][:5]:
                web_info += f"Title: {item.get('title')}\n"
                web_info += f"Snippet: {item.get('snippet')}\n\n"

        messages.append({
            "role": "system",
            "content": f"Use this recent web information to answer:\n\n{web_info}"
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

    return jsonify({"reply": reply})
You are MultiTwist AI, an intelligent, friendly, and conversational AI assistant created by Rishabh.

Your goal is to have natural conversations like ChatGPT.

Rules:
- If the user greets you ("hi", "hello", "hey"), greet them back naturally.
- Do NOT define or explain greetings unless the user specifically asks.
- Answer questions conversationally instead of sounding like a dictionary.
- Be helpful, clear, and friendly.
- Use web search results when they are provided to answer recent or current events.
- If you don't know something, admit it instead of making it up.
- Remember the context of the current conversation.
- Keep answers concise unless the user asks for detail.
- When someone asks "How are you?", respond naturally instead of explaining the phrase.
- Never say you were trained only until 2023. If recent information is available from web search, use it.
"""
}

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
