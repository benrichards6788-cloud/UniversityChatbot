from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return """
    <h1>Policy Chatbot</h1>
    <form method="post" action="/ask">
        <input type="text" name="question" placeholder="Ask a question" style="width: 300px;" />
        <button type="submit">Ask</button>
    </form>
    """

@app.route("/ask", methods=["POST"])
def ask():
    question = request.form.get("question") or request.json.get("question")

    # TEMP TEST RESPONSE
    answer = f"You asked: {question}"

    if request.is_json:
        return jsonify({"answer": answer})

    return f"""
    <h1>Answer</h1>
    <p><strong>Question:</strong> {question}</p>
    <p><strong>Answer:</strong> {answer}</p>
    <p><a href="/">Ask another question</a></p>
    """