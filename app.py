from flask import Flask, send_file, request, jsonify

app = Flask(**name**)

@app.route("/")
def home():
return send_file("multitwist.html")

@app.route("/chat", methods=["POST"])
def chat():
data = request.json
message = data["message"]

```
reply = f"You said: {message}"

return jsonify({"reply": reply})
```

if **name** == "**main**":
app.run()
