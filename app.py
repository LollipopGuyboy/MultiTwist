from flask import Flask, send_file, request, jsonify
import ollama

app = Flask(__name__)

@app.route("/")
def home():
    return send_file("multitwist.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    message = data["message"]

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": message
            }
        ]
    )

    reply = response["message"]["content"]

    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(debug=True)