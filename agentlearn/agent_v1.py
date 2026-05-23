# 国内网络：用 HuggingFace 镜像，否则下载模型会超时
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ============================================================
# Agent 学习路线
# ============================================================
# 第一课：从固定 Chain 到自主 Agent —— LLM 自主选工具
# 第二课：多工具协配 —— 本地检索 + 网络搜索 + 图片OCR
# 第三课：对话记忆 —— Agent 记住之前聊过什么
# 第四课：流式输出 —— 实时看 Agent 每一步动作（本节）
# ============================================================

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool           # ★ 新概念：@tool 装饰器
from langchain.agents import create_agent       # ★ 新概念：创建 Agent
from langgraph.checkpoint.memory import InMemorySaver  # ★ 新概念：对话记忆
from langchain_community.tools.tavily_search import TavilySearchResults  # ★ 新工具：网络搜索

# ---------- 第①步：摄入（和之前一模一样，你学过了） ----------
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-zh-v1.5")

llm = ChatOpenAI(
    api_key="sk-63112b3e2de441fd9a2f4a8a3ab75753",
    base_url="https://api.deepseek.com",
    model="deepseek-v4-pro",
    max_tokens=4096,  # Agent 回答可能很长（表格等），默认值容易截断
    extra_body={"thinking": {"type": "disabled"}},  # 禁用思考模式，否则 Agent 多轮调用会报错
)

# 确保从脚本所在目录找文件（这样在哪里运行都不会出错）
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "example.pdf")
loader = PyPDFLoader(file_path) if file_path.endswith(".pdf") else None
documents = loader.load()
chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(documents)
vectorstore = Chroma.from_documents(chunks, embeddings, collection_name="agent_v1_collection")

# ---------- 第②步：把能力包装成工具 ----------
# 工具1：本地文档检索（你已经熟悉的）

retriever = vectorstore.as_retriever(search_kwargs={"k": 6})


@tool
def search_local_docs(query: str) -> str:
    """
    搜索本地已上传的财务文档（公司报告、财务报表、合同等）。
    当用户询问关于公司内部数据、历史报表、合同条款、折旧政策、
    报销标准等本地文档中可能包含的信息时，必须使用此工具。
    """
    docs = retriever.invoke(query)
    if not docs:
        return "未在本地文档中找到相关内容。"
    return "\n\n---\n\n".join(
        f"[来源: {doc.metadata.get('source', '未知')}]\n{doc.page_content}"
        for doc in docs
    )





# 工具3：图片 OCR 识别（新增）
# Agent 不懂看图，但工具可以"替它看"——读到文字后交还给 Agent 理解
# 这就是"大脑（LLM）+ 感官（工具）"的架构

import easyocr
_ocr = easyocr.Reader(["ch_sim", "en"], gpu=False)  # 首次运行会下载模型，稍等

@tool
def analyze_image(image_path: str) -> str:
    """
    识别图片中的文字（OCR）。
    当用户上传了图片文件（jpg/png），或询问扫描件、发票照片中的内容时，使用此工具。
    传入图片文件的完整路径，返回图片中提取到的文字。
    """
    try:
        results = _ocr.readtext(image_path)
        if not results:
            return "图片中未识别到文字，可能是纯图或手写内容。"
        lines = [f"  {text}  (置信度: {conf:.0%})" for (_, text, conf) in results]
        return "[从图片中识别的文字]:\n" + "\n".join(lines)
    except FileNotFoundError:
        return f"找不到图片文件: {image_path}"
    except Exception as e:
        return f"图片识别失败（{type(e).__name__}）: {str(e)}"
# 工具2：互联网搜索
# 用 @tool 把 Tavily 包一层：出错时 Agent 能收到提示，不会直接崩溃
TAVILY_API_KEY = "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"

_tavily = TavilySearchResults(
    tavily_api_key=TAVILY_API_KEY,
    max_results=5,
    search_depth="advanced",
)


@tool
def search_web(query: str) -> str:
    """
    搜索互联网获取最新公开信息。
    当用户询问最新政策法规、行业动态、市场数据、新闻，
    或本地文档中未涵盖的信息时，使用此工具。
    """
    try:
        return _tavily.invoke(query)
    except Exception as e:
        return f"网络搜索失败（{type(e).__name__}）。请基于本地已有知识回答用户，并建议用户检查网络代理。"


# ---------- 第③步：Agent 的"大脑"（系统提示词） ----------
# 现在有两个工具了，必须教会 Agent 什么场景用哪个

SYSTEM_PROMPT = """你是一个严谨的财务分析助手，拥有三个工具：

1. search_local_docs —— 搜索本地已上传的财务文档
2. search_web —— 搜索互联网获取最新公开信息
3. analyze_image —— 识别图片中的文字（OCR）

选择工具的规则：
- 公司内部数据、历史报销、合同条款 → search_local_docs
- 最新政策、税法变动、行业趋势 → search_web
- 用户提到图片文件（jpg/png）、扫描件、发票照片 → analyze_image
- 不确定时多工具配合，综合回答
- 闲聊直接回答，不调工具
- 严格基于工具结果回答，找不到就如实说
- 用中文回答，清晰有条理"""

# ---------- 第④步：创建 Agent（配 3 个工具 + 记忆） ----------
# checkpointer=InMemorySaver() 让 Agent 记住所有对话历史
# 每次 invoke 时用同一个 thread_id，它就记住了

agent = create_agent(
    llm,
    [search_local_docs, search_web, analyze_image],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=InMemorySaver(),  # ★ 这行就是记忆
)

# ---------- 第⑤步：交互（流式输出 - messages 模式） ----------
# stream_mode="messages" → LLM 每生成几个字就立刻返回，真正的打字机效果
# 对比 updates：updates 等整个节点跑完才返回一块
# messages：token 级别的流式，像 ChatGPT 那样一个字一个字往外蹦

print("=" * 60)
print("Agent 已就绪，配有 3 个工具 + 对话记忆 + 流式输出（输入 '退出' 结束）")
print("  工具1: search_local_docs  — 搜本地财务文档")
print("  工具2: search_web         — 搜互联网最新信息")
print("  工具3: analyze_image      — 识别图片中的文字(OCR)")
print()
print("试试连续问（Agent 会记住上文）：")
print("  你: '乙材料入账价值是多少？'")
print("  你: '那折旧政策呢？'  ← Agent 理解你在问乙材料的折旧")
print("=" * 60)

while True:
    question = input("\n请输入你的问题：")
    if question == "退出":
        print("Bye!")
        break

    print()  # 空一行

    try:
        for msg, metadata in agent.stream(
            {"messages": [{"role": "user", "content": question}]},
            config={"configurable": {"thread_id": "conversation_1"}},
            stream_mode="messages",
        ):
            node_name = metadata.get("langgraph_node", "")

            if node_name in ("agent", "model"):
                # 检查是否包含完整的工具调用（有 name 的才是完整 chunk）
                msg_tool_calls = getattr(msg, "tool_calls", None) or []
                real_calls = [
                    tc for tc in msg_tool_calls
                    if (isinstance(tc, dict) and tc.get("name"))
                    or (hasattr(tc, "name") and tc.name)
                ]

                if real_calls:
                    print()  # 换行，和之前的文字分开
                    for tc in real_calls:
                        name = tc.get("name") if isinstance(tc, dict) else tc.name
                        args = tc.get("args") if isinstance(tc, dict) else tc.args
                        print(f"🔧 调用工具: {name}")
                        print(f"   参数: {args}")

                # 文字内容 — 逐字输出（打字机效果的核心）
                msg_content = getattr(msg, "content", "")
                if msg_content:
                    if isinstance(msg_content, str):
                        print(msg_content, end="", flush=True)
                    elif isinstance(msg_content, list):
                        for part in msg_content:
                            text = part.get("text", "") if isinstance(part, dict) else str(part)
                            print(text, end="", flush=True)

            elif node_name == "tools":
                content = getattr(msg, "content", "")
                preview = str(content)[:200]
                more = "..." if len(str(content)) > 200 else ""
                print(f"\n📋 工具返回: {preview}{more}\n")

    except Exception as e:
        print(f"\n❌ 出错了: {type(e).__name__}: {e}")

    print()  # 回答后空一行
