from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from configparser import ConfigParser
import os

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("Jabra-Talk25.pdf")
data = loader.load()
loader2 = PyPDFLoader("Jabra-Talk45.pdf")
data2 = loader2.load()

data.extend(data2)
config = ConfigParser()
config.read("config.ini")

os.environ["GOOGLE_API_KEY"]=config["Gemini"]["API_KEY"]
from langchain.embeddings import SentenceTransformerEmbeddings

embeddings = SentenceTransformerEmbeddings(model_name="distiluse-base-multilingual-cased-v1")
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(data)
db = FAISS.from_documents(docs, embeddings)

query = "請列出Jabra Talk25與Jabra Talk45分別是為什麼而打造的？"
results = db.similarity_search_with_score(query, 2)
print(results[0][0].page_content)
print(results[1][0].page_content)

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    """Answer the following question based only on the provided context:
    <context>
    {context}
    </context>
    Question:{input}"""
)

from langchain.chains.combine_documents import create_stuff_documents_chain
document_chain = create_stuff_documents_chain(llm, prompt)
#query = input("請輸入您的問題？")
#query="請問這兩個產品的性價比如何？"
query="這兩個產品的休眠待機時間多長？我要去東南亞出差一個禮拜帶哪台好"
#query="Jabra Talk 25與Jabra Talk 45規格差異是什麼，請以Table列出規格差異"
results = db.similarity_search_with_score(query,2)
print("Retrieved related content:")
print(results[0][0].page_content)
print("==================================")

llm_result = document_chain.invoke(
    {
        "input":query,
        "context":[results[0][0], results[1][0]],
    }
)

print("Question:", query)
print("LLM Answer:", llm_result)