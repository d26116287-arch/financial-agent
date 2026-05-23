from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

#embeddings模型接入
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-zh-v1.5")

#接入deepseek llm api
llm = ChatOpenAI(api_key="sk-63112b3e2de441fd9a2f4a8a3ab75753"
, base_url="https://api.deepseek.com"
,model="deepseek-v4-pro")

#自动识别文件类型（支持 txt/pdf）
file_path = "example.pdf"
if file_path.endswith(".pdf"):
    loader = PyPDFLoader(file_path)
else:
    loader = TextLoader(file_path, encoding="utf-8")
documents = loader.load()
chunks=RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(documents)

#向量存入库
vectorstore=Chroma.from_documents(chunks, embeddings, collection_name="my_collection")

#prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个严谨的财务助手。请严格根据提供的【参考资料】回答。不要瞎编。\n【参考资料】：\n{context}"),
    ("user", "{question}")   # {question} 是占位符，调用时填入
])

# 组装链：先检索 → 再问答
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
def build_context(question):
    docs = retriever.invoke(question)
    return "\n\n".join(doc.page_content for doc in docs)

#连接一下
chain=prompt | llm    #它最大的魅力在于：以后哪怕你想加入“翻译成英文”、“核对计算公式”、“把结果保存进数据库”等新步骤，只需要像拼乐高一样在后面继续加竖线（比如 chain = prompt | llm | translator | database）就行了，完全不需要去改动前面错综复杂的代码。
my_question = input("请输入你的问题：")
context = build_context(my_question)
message = chain.invoke({"context": context, "question": my_question})
print("\n=== LangChain 提取结果 ===")
print(message.content)




#为什么 RAG 是“省钱大师”？
#这就是我们为什么要学 切片（Chunking） 和 检索（Retrieval） 的原因。

#当你问“LED 大屏折旧几年”时，RAG 系统并不会把整个 200 页的文档发给 DeepSeek，它只会：

##在本地数据库（Chroma）里找出包含“LED”、“折旧”、“入账价值”的 3-5 个小片段（总共可能不到 1000 字）。

#只把这 1000 字 和你的问题发给 DeepSeek。

#这样，你实际消耗的 Token 只有 几百到一两千个，成本瞬间降低了 1000 倍！