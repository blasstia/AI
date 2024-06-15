import requests
from IPython.display import Image
#image_url = "http://t2.gstatic.com/licensed-image?q=tbn:ANd9GcRcAIa6SlcR0AihK2UQmbgKKTmMbjeOX6orm_hXl7xW2DZNWS4q-4i-G7IlErM8egImT-Mm6AfBV1vYucl51zg"
image_url = "cat.jpg"
#content = requests.get(image_url).content
#Image(content)
from configparser import ConfigParser
import os

# Set up config parser
config = ConfigParser()
config.read("config.ini")
# Set up Google Gemini API key
os.environ["GOOGLE_API_KEY"] = config["Gemini"]["API_KEY"]

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
# example
# image_url = "images.jpeg"
image_url_2 = "dog.jpg"
image_url_3 = "llama.jpg"
#user_question = "請問這兩種動物可不可以當好朋友？"
user_question = "請問這三種動物打架誰會贏？"
user_chinese = " 請使用繁體中文回答。"
message = HumanMessage(
    content=[
        {
            "type": "text",
            "text": user_question + user_chinese,
        },  # You can optionally provide text parts
        {"type": "image_url", "image_url": image_url},
        {"type": "image_url", "image_url": image_url_2},
        {"type": "image_url", "image_url": image_url_3}
    ]
)
result = llm.invoke([message])
print("問：", user_question)
print("答：", result.content.lstrip(" "))
if "http" in image_url:
    content = requests.get(image_url).content
else:
    content = image_url
Image(content)