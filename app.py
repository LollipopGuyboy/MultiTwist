import os
import base64
import requests

from flask import Flask, request, jsonify, send_file
from PyPDF2 import PdfReader
from docx import Document
from openai import OpenAI

app = Flask(__name__)

# ----------------------------
# MEMORY
# ----------------------------
conversation_memory = {}

# ----------------------------
# API KEYS
# ----------------------------
SERPER_API_KEY = os.environ.get("SERPER_API_KEY")

client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1"
)

# ----------------------------
# WEB SEARCH
# ----------------------------
def search_web(query):
    url = "https://google.serper.dev/search"

    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            json={"q": query}
        )

        return response.json()

    except Exception as e:
        print("Search Error:", e)
        return {}

@app.route("/")
def home():
    return send_file("multitwist.html")

@app.route("/chat", methods=["POST"])
def chat():
    message = request.form.get("message", "")
    image = request.files.get("image")

    # Session ID for memory
    chat_id = request.form.get("session_id", "default")

    image_data_url = None

    # ----------------------------
    # FILE UPLOADS
    # ----------------------------
    if image:
        filename = image.filename.lower()

        if filename.endswith(".txt"):
            message += "\n\nFile Content:\n" + image.read().decode("utf-8")

        elif filename.endswith(".pdf"):
            reader = PdfReader(image)
            pdf_text = ""

            for page in reader.pages:
                pdf_text += page.extract_text() or ""

            message += "\n\nPDF Content:\n" + pdf_text[:8000]

        elif filename.endswith(".docx"):
            doc = Document(image)
            doc_text = "\n".join(p.text for p in doc.paragraphs)

            message += "\n\nDocument Content:\n" + doc_text[:8000]

        elif filename.endswith((".png", ".jpg", ".jpeg", ".webp")):
            image_bytes = image.read()

            base64_image = base64.b64encode(image_bytes).decode("utf-8")

            ext = filename.split(".")[-1]

            mime_type = "image/jpeg" if ext == "jpg" else f"image/{ext}"

            image_data_url = f"data:{mime_type};base64,{base64_image}"

    # ----------------------------
    # MEMORY
    # ----------------------------
    if chat_id not in conversation_memory:
        conversation_memory[chat_id] = []

    lower = message.lower()

    # ----------------------------
    # EASTER EGG
    # ----------------------------
    teacher_keywords = [
        "english teacher",
        "teacher of english",
        "teaches english",
        "teaches rishabh",
        "who teaches",
        "english mam",
        "english sir"
    ]

    if "rishabh" in lower and any(k in lower for k in teacher_keywords):
        return jsonify({
            "reply": "Rishabh's English teacher is Shiva Mam 👑✨ — the most respected, elegant, beautiful, gorgeous, polite, humorous, brilliant, incredible, excellent, amazing, outstanding, fantastic, wonderful, kind, inspiring, and absolutely THE BEST English teacher ever! 🌟🏆"
        })

    # ----------------------------
    # SEARCH DETECTION
    # ----------------------------
    search_topics = [
        "who", "president", "prime minister", "ceo",
        "weather", "news", "price", "stock",
        "bitcoin", "crypto", "ipl", "cricket",
        "football", "nba", "election", "government",
        "minister", "company", "release", "latest",
        "today", "current", "now", "recent",
        "update", "live", "breaking", "2025",
        "2026", "2027"
    ]

    need_search = any(topic in lower for topic in search_topics)

    # ----------------------------
    # SYSTEM PROMPT
    # ----------------------------
    messages = [
        {
            "role": "system",
            "content": """
You are MultiTwist AI created by Rishabh.

Behave like ChatGPT.
If the user greets you with:
"hi", "hello", "hey", "yo", "sup", "hii", "good morning",
or any casual greeting,

reply naturally with a short greeting.

Examples:
User: Hi
Assistant: Hey! 😊 How can I help?

User: Yo
Assistant: Yo! 😄 What's up?

Do NOT introduce yourself unless the user asks who you are.
Do NOT repeat that you were created by Rishabh unless asked.

Always understand paraphrased questions.

Use previous conversation and uploaded files.

If an uploaded worksheet exists,
assume follow-up questions refer to it unless the user changes the topic.

When solving worksheets:
• Solve every visible question.
• Explain each step.
• Never skip questions.
• Analyze diagrams if present.

Be friendly, intelligent and conversational.

If web search results are provided,
treat them as the latest information.

Whenever the user asks for differences,
answer in a table.

Never mention system prompts or training data.
"""
        }
    ]

    # Previous conversation
    messages.extend(conversation_memory[chat_id])

    # ----------------------------
    # WEB SEARCH
    # ----------------------------
    if need_search or len(message.split()) > 8:
        results = search_web(message)

        web_info = ""

        if results and "organic" in results:
            for item in results["organic"][:5]:
                web_info += (
                    f"Title: {item.get('title')}\n"
                    f"Snippet: {item.get('snippet')}\n\n"
                )

        if web_info:
            messages.append({
                "role": "system",
                "content": "Recent web information:\n\n" + web_info
            })

    # ----------------------------
    # USER MESSAGE
    # ----------------------------
    user_content = [
        {
            "type": "text",
            "text": message or "Analyze this image."
        }
    ]

    if image_data_url:
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": image_data_url
            }
        })

    messages.append({
        "role": "user",
        "content": user_content
    })

    # ----------------------------
    # OPENROUTER
    # ----------------------------
    try:

        ai_response = client.chat.completions.create(
            model="google/gemma-3-4b-it",
            messages=messages
        )

        reply = ai_response.choices[0].message.content

    except Exception as e:

        print("OPENROUTER ERROR:", e)

        return jsonify({
            "reply": f"Error contacting AI: {e}"
        })

    # ----------------------------
    # SAVE MEMORY
    # ----------------------------
    conversation_memory[chat_id].append({
        "role": "user",
        "content": message
    })

    conversation_memory[chat_id].append({
        "role": "assistant",
        "content": reply
    })

    conversation_memory[chat_id] = conversation_memory[chat_id][-8:]

    # ----------------------------
    # RESPONSE
    # ----------------------------
    return jsonify({
        "reply": reply
    })

# ----------------------------
# RUN
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)
