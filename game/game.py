from flask import Flask, render_template, url_for
from flask import request
from configparser import ConfigParser
import os
import google.generativeai as genai


# Config Parser
config = ConfigParser()
config.read("config.ini")

genai.configure(api_key=config["Gemini"]["API_KEY"])

# os.environ["GOOGLE_API_KEY"] = config["Gemini"]["API_KEY"]
# from langchain_google_genai import ChatGoogleGenerativeAI
# llm = ChatGoogleGenerativeAI(model="gemini-pro")
from google.generativeai.types import HarmCategory, HarmBlockThreshold

llm = genai.GenerativeModel(
    "gemini-1.5-flash-latest",
    safety_settings={
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    },
    generation_config={
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
    },
)


app = Flask(__name__)


@app.route("/")
def home():
    return render_template("new.html")


@app.route("/call_llm", methods=["POST"])
def call_llm():
    if request.method == "POST":
        print("POST!")
        data = request.form
        print(data)
        # result = llm.invoke("請講一句打招呼的話，不超過 20 字")
        result = llm.generate_content("請講一句凶狠的話，不超過 20 字")
        return result.text
