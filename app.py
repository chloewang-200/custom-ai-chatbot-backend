from requests.exceptions import ConnectionError
from services.huggingFace import HuggingFace
from services.stabilityAI import StabilityAI
from services.custom import Custom
from openAI import OpenAI
from services.cohere import Cohere
from flask import Flask, session, request, jsonify
from flask_session import Session
from dotenv import load_dotenv
from flask_cors import CORS
import os

# ------------------ SETUP ------------------

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# this will need to be reconfigured before taking the app to production
cors = CORS(app, supports_credentials=True)

@app.route('/')
def home():
    return 'Hello, World!'

# ------------------ EXCEPTION HANDLERS ------------------

# Sends response back to Deep Chat using the Response format:
# https://deepchat.dev/docs/connect/#Response
@app.errorhandler(Exception)
def handle_exception(e):
    print(e)

    return {"error": str(e)}, 500

@app.errorhandler(ConnectionError)
def handle_exception(e):
    print(e)
    return {"error": "Internal service error"}, 500


# ------------------ OPENAI API ------------------

open_ai = OpenAI()

@app.route("/openai-chat", methods=["POST"])
def openai_chat():
    body = request.json
    if 'conversation_history' not in session:
        session['conversation_history'] = []
    session['conversation_history'].append("test")
    session.modified = True
    print("received chat request", session['conversation_history'], session.sid)
    return open_ai.chat(body)

@app.route("/openai-chat-stream", methods=["POST"])
def openai_chat_stream():
    body = request.json
    return open_ai.chat_stream(body)

@app.route("/openai-image", methods=["POST"])
def openai_image():
    files = request.files.getlist("files")
    return open_ai.image_variation(files)

# ------------------ START SERVER ------------------

if __name__ == "__main__":
    print("running")
    app.run(port=8080, debug=True)
