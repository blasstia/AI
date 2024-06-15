import requests
from IPython.display import Image, display
from matplotlib import pyplot as plt
from io import BytesIO
import PIL
from configparser import ConfigParser
import os
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

image_url = "stock.png"

# Set up config parser
config = ConfigParser()
config.read("config.ini")
# Set up Google Gemini API key
os.environ["GOOGLE_API_KEY"] = config["Gemini"]["API_KEY"]

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
# example
# image_url = "images.jpeg"
image_url_1 = "tsmc.jpg"
image_url_2 = "umc.jpg"
image_url_3 = "cht.jpg"
images = [
    image_url,
    image_url_1,
    image_url_2,
    image_url_3,
]
user_question = "圖片中的表格就是我的投資，請依據另外三張圖片呈現的最新股票資訊，計算出當前損益狀況。"
user_chinese = " 請使用繁體中文回答。"
message = HumanMessage(
    content=[
        {
            "type": "text",
            "text": user_question + user_chinese,
        },  # You can optionally provide text parts
        {"type": "image_url", "image_url": image_url},
        {"type": "image_url", "image_url": image_url_1},
        {"type": "image_url", "image_url": image_url_2},
        {"type": "image_url", "image_url": image_url_3}
    ]
)
result = llm.invoke([message])
print("問：", user_question)
print("答：", result.content.lstrip(" "))
for image_url in images:
    if "http" in image_url:
        content = requests.get(image_url).content
    else:
        content = image_url
    display(Image(content))