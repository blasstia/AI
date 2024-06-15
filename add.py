from configparser import ConfigParser
from langchain_google_genai import ChatGoogleGenerativeAI
import os

config = ConfigParser()
config.read("config.ini")

os.environ["GOOGLE_API_KEY"]=config["Gemini"]["API_KEY"]

llm = ChatGoogleGenerativeAI(model="gemini-pro")
result = llm.invoke("中壢景點")
print(result.content)

from langchain_core.messages import HumanMessage, SystemMessage
role_play = "你是一個新加坡人，口頭禪是好犀利"
user_input = input("請輸入您要的問題：")
model = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True)
result = model.invoke(role_play + user_input)
print(result.content)