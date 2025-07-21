from flask import Flask, request, jsonify
from flask_cors import CORS
from question_selector import get_top_questions

app = Flask(__name__)
CORS(app)

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    narrative = data.get("narrative", "")

    if not narrative:
        return jsonify({"error": "Missing 'summary' in request body"}), 400
    
    questions = get_top_questions(narrative)
    
    return jsonify({
        "questions": questions
    })
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)