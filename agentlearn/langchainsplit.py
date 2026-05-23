from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

api_key = "sk-63112b3e2de441fd9a2f4a8a3ab75753"
base_url = "https://api.deepseek.com"

# 本地 embedding，免费，中文友好
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

loader = TextLoader("company_report.txt", encoding="utf-8")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
all_splits = text_splitter.split_documents(data)

print("正在计算向量并建立本地向量库...")
vectorstore = FAISS.from_documents(documents=all_splits, embedding=embeddings)
vectorstore.save_local("my_faiss_index")
print("本地向量库已存入 my_faiss_index 文件夹")

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(api_key=api_key, base_url=base_url, model="deepseek-chat"),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

query = "帮我查一下去苏大招聘的报销标准，以及乙材料的入账价值。"
print(f"\n用户提问：{query}")
response = qa_chain.invoke(query)

print("\n=== AI 检索后的回答 ===")
print(response["result"])
