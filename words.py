from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from configparser import ConfigParser
import os

# Get word data by Docx2txtLoader
from langchain_community.document_loaders import Docx2txtLoader

loader = Docx2txtLoader("rent_contract.docx")
data = loader.load()

# Set up config parser
config = ConfigParser()
config.read("config.ini")

# Set up Google Gemini API key
os.environ["GOOGLE_API_KEY"] = config["Gemini"]["API_KEY"]

from langchain.embeddings import SentenceTransformerEmbeddings
# need takes 539MB
embeddings = SentenceTransformerEmbeddings(model_name="distiluse-base-multilingual-cased-v1")
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(data)
db = FAISS.from_documents(docs, embeddings)
query = "如果我想終止租約，我應該要多久以前通知房東？"
results = db.similarity_search_with_score(query, 1)
print(results[0][0].page_content)
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-pro")

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
"""
你是一名房東
Answer the following question based only on the provided context:
<context>
{context}
</context>
Question: {input}
請用繁體中文作答
"""
)

from langchain.chains.combine_documents import create_stuff_documents_chain

document_chain = create_stuff_documents_chain(llm, prompt)

# 如果我想終止租約，我應該要多久以前通知房東？
# 如果我想終止租約，需要什麼證件？
query = input("請輸入您的問題: ")
results = db.similarity_search_with_score(query, 1)
print("取得相關的解答 :")
print(results[0][0].page_content)
print("====================================================")

llm_result = document_chain.invoke(
    {
        "input": query,
        "context": [results[0][0]],
    }
)

print("Question:", query)
print("LLM Answer:", llm_result.lstrip(" "))