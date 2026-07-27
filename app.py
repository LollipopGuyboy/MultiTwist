from flask import Flask, send_file, request, jsonify
from groq import Groq
import requests
import os

app = Flask(__name__)

# ----------------------------
# MEMORY
# ----------------------------

conversation_memory = {}

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
    chat_id = data.get("chat_id", "default")

    if chat_id not in conversation_memory:
        conversation_memory[chat_id] = []

    lower = message.lower()

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
        "breaking",
        "update",
        "who won"
    ]

    need_search = any(word in lower for word in search_words)

    messages = [
        {
            "role": "system",
            "content": """
You are MultiTwist AI created by Rishabh.

Behave like ChatGPT.

Rules:
- Be friendly.
- Talk naturally.
- If the user says hello, hi, hey, greet them normally.
- Don't explain greetings.
- Remember previous messages in the conversation.
- Only use web information if it is provided.
"""
        }
    ]

    # Previous conversation
    messages.extend(conversation_memory[chat_id])

    # Web search if needed
    if need_search:

        results = search_web(message)

        web_info = ""

        if "organic" in results:

            for item in results["organic"][:5]:

                web_info += f"Title: {item.get('title')}\n"
                web_info += f"Snippet: {item.get('snippet')}\n\n"

        messages.append({
            "role": "system",
            "content": "Recent web information:\n\n" + web_info
        })

    # Current message
    messages.append({
        "role": "user",
        "content": message
    })

    # Ask Groq
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages
    )

    reply = response.choices[0].message.content

    # Save memory
    conversation_memory[chat_id].append({
        "role": "user",
        "content": message
    })

    conversation_memory[chat_id].append({
        "role": "assistant",
        "content": reply
    })

    # Keep only last 20 messages
    conversation_memory[chat_id] = conversation_memory[chat_id][-20:]

    return jsonify({
        "reply": reply
    })

# ----------------------------
# RUN
# ----------------------------

if __name__ == "__main__":
    app.run(debug=True)
