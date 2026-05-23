from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
#接入api
llm = ChatOpenAI(api_key="sk-63112b3e2de441fd9a2f4a8a3ab75753"
, base_url="https://api.deepseek.com"
,model="deepseek-v4-pro")
#读取文件 txt  如果pdf再改
loader = TextLoader("example.txt", encoding="utf-8")
documents = loader.load()
file_content = documents[0].page_content
#prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个严谨的财务助手。请严格根据提供的【参考资料】回答。不要瞎编。\n【参考资料】：\n{context}"),
    ("user", "{question}")   # {question} 是占位符，调用时填入
])
#连接一下
chain=prompt | llm    #它最大的魅力在于：以后哪怕你想加入“翻译成英文”、“核对计算公式”、“把结果保存进数据库”等新步骤，只需要像拼乐高一样在后面继续加竖线（比如 chain = prompt | llm | translator | database）就行了，完全不需要去改动前面错综复杂的代码。
my_question = input("请输入你的问题：")
message = chain.invoke({"context": file_content, "question": my_question})
print("\n=== LangChain 提取结果 ===")
print(message.content)