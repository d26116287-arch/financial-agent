from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 1. 嵌入模型
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

# 2. LLM
llm = ChatOpenAI(
    api_key="sk-63112b3e2de441fd9a2f4a8a3ab75753",
    base_url="https://api.deepseek.com",
    model="deepseek-v4-pro",
)

# 3. 加载文档并切片
loader = TextLoader("example.txt", encoding="utf-8")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# 4. 向量入库
vectorstore = Chroma.from_documents(chunks, embeddings, collection_name="my_collection")

# 5. Prompt 模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个严谨的财务助手。请严格根据提供的【参考资料】回答。不要瞎编。\n【参考资料】：\n{context}"),
    ("user", "{question}"),
])

# 6. 组装链：先检索 → 再问答
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


def build_context(question):
    docs = retriever.invoke(question)
    return "\n\n".join(doc.page_content for doc in docs)


# 7. 交互
my_question = input("请输入你的问题：")
context = build_context(my_question)
message = llm.invoke(prompt.format(context=context, question=my_question))
print("\n=== LangChain 提取结果 ===")
print(message.content)
