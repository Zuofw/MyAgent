"""
这是一个基于LangChain的多代理系统，包含：
1. 技术代理：处理技术问题，可以搜索网络和文档
2. 数学代理：处理数学计算问题
3. 路由器代理：判断用户输入的问题类型并决定调用哪个代理
"""

# 导入标准库
from tabnanny import verbose
import os
from dotenv import load_dotenv  # 用于加载环境变量

# 导入LangChain相关组件
# 代理创建和执行相关
from langchain_classic.agents import create_openai_tools_agent, AgentExecutor
# 链式调用相关
from langchain_classic.chains.llm import LLMChain
# 搜索工具
from langchain_tavily import TavilySearch
# 向量存储相关
from langchain_community.vectorstores import FAISS
# 可运行对象包装器，用于添加对话历史
from langchain_core.runnables import RunnableWithMessageHistory
# 工具创建相关
from langchain_core.tools import create_retriever_tool, tool
# 文本分割器
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 文档加载器
from docutils.nodes import document
from langchain_community.document_loaders import TextLoader, WebBaseLoader
# 嵌入模型（使用HuggingFace本地模型）
from langchain_huggingface import HuggingFaceEmbeddings
# OpenAI聊天模型
from langchain_openai import ChatOpenAI
# 消息历史记录
from langchain_community.chat_message_histories import ChatMessageHistory
# 提示模板
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder

# 全局变量存储代理实例
technical_agent_instance = None
math_agent_instance = None

def create_first_agent():
    """
    创建技术代理，用于处理技术相关问题
    这个代理可以：
    1. 搜索网络获取最新信息
    2. 检索特定文档（如LangSmith文档）
    3. 进行对话
    """
    # 创建一个消息历史对象，用于存储对话历史
    message_history = ChatMessageHistory()
    
    # 创建一个文档加载器，从指定URL加载文档内容
    # 这里加载的是LangSmith的文档，作为知识库
    loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
    docs = loader.load()
    
    # 创建一个文档分割器，将长文档分割成小块
    # chunk_size=1000: 每块最大1000个字符
    # chunk_overlap=200: 块之间重叠200个字符，确保上下文连贯
    documents = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    ).split_documents(docs)
    
    # 使用本地嵌入模型替代 OpenAI embeddings
    # 原因：用户的API端点不支持embeddings功能，所以使用本地模型
    # all-MiniLM-L6-v2: 一个轻量级但效果不错的嵌入模型
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 创建一个向量数据库,参数是document和embeddings,代表文档和嵌入模型
    vector = FAISS.from_documents(documents, embeddings)
    
    # 创建一个向量数据库检索器，用于后续的文档检索
    retriever = vector.as_retriever()
    
    # 创建一个向量数据库检索器工具
    # 这个工具允许代理在需要时搜索LangSmith相关文档
    retriever_tool = create_retriever_tool(
        retriever,
        "langsmith_search",  # 工具名称
        "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",  # 工具描述
    )

    # 检查是否有Tavily API密钥，如果有则添加搜索工具
    tools = [retriever_tool]
    
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if tavily_api_key:
        # 使用Tavily搜索工具，允许代理搜索网络获取最新信息
        search = TavilySearch(tavily_api_key=tavily_api_key)
        tools.append(search)
    else:
        print("警告: 未找到TAVILY_API_KEY，跳过网络搜索功能")

    # 创建聊天模型实例
    # 使用用户在.env文件中配置的API端点和密钥
    llm = ChatOpenAI(
        model="gpt-3.5-turbo:free",  # 使用的模型
        base_url=os.getenv("OPENAI_BASE_URL"),  # API基础URL
        api_key=os.getenv("OPENAI_API_KEY"),  # API密钥
        temperature=0  # 温度参数，0表示输出更确定性，不随机
    )

    # 创建提示模板
    # 包含系统提示、用户输入和代理执行步骤的占位符
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),  # 系统提示
            ("human", "{input}"),  # 用户输入占位符
            MessagesPlaceholder(variable_name="agent_scratchpad")  # 代理执行步骤占位符
        ]
    )
    
    # 创建一个代理，将LLM、工具和提示模板组合在一起
    agent = create_openai_tools_agent(llm, tools, prompt)
    
    # 创建一个代理执行器，负责执行代理并处理工具调用
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    # 为代理添加对话历史记忆能力
    # RunnableWithMessageHistory: 包装代理执行器，添加历史记录功能
    # lambda session_id: message_history: 会话历史记录存储
    # input_messages_key: 指定输入中用户消息的键名
    # history_messages_key: 指定历史记录在提示中的键名
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        lambda session_id: message_history,
        input_messages_key="input",
        history_messages_key='chat_history',
    )
    return agent_with_chat_history

# 创建一个基础计算器工具
@tool
def basic_calculator(query):
    """
    基础计算器工具
    使用eval函数计算数学表达式
    """
    try:
        # 使用eval函数计算表达式（注意：在生产环境中应避免使用eval）
        result = eval(query)
        return f"The result is {result}"
    except (SyntaxError, NameError) as e:
        # 捕获语法错误或名称错误
        return f"Sorry, I couldn't calculate that due to an error: {e}"


# 创建一个方程求解工具
@tool
def equation_solver(query):
    """
    方程求解工具（占位符）
    当前版本只是占位符，未来可以实现具体的方程求解逻辑
    """
    # Basic equation solver (placeholder)
    # Implement specific logic for solving equations
    return "Equation solver: This feature is under development."


# 创建一个调用技术代理的工具
@tool
def invoke_technical_agent(query: str) -> str:
    """
    调用技术代理工具
    """
    global technical_agent_instance
    if technical_agent_instance is None:
        technical_agent_instance = create_first_agent()
        
    result = technical_agent_instance.invoke(
        {"input": query},
        config={"configurable": {"session_id": "test"}},
    )
    return result["output"]

# 创建一个调用数学代理的工具
@tool
def invoke_math_agent(query: str) -> str:
    """
    调用数学代理工具
    """
    global math_agent_instance
    if math_agent_instance is None:
        math_agent_instance = create_second_agent()
        
    result = math_agent_instance.invoke(
        {"input": query},
        config={"configurable": {"session_id": "test"}},
    )
    return result["output"]

def create_second_agent():
    """
    创建数学代理，专门处理数学相关问题
    这个代理配备了计算器和方程求解工具
    """
    # 创建聊天模型实例，配置与技术代理相同
    llm = ChatOpenAI(
        model="gpt-4o-mini:free",  # 使用不同的模型
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0  # 保持输出确定性
    )

    # 为数学代理配置工具
    tools = [basic_calculator, equation_solver]

    # 创建专门用于数学问题的提示模板
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你很擅长解决与数学相关的问题。请将问题的答案返回给用户"),  # 中文系统提示
            ("human", "{input}"),  # 用户输入占位符
            MessagesPlaceholder(variable_name="agent_scratchpad")  # 代理执行步骤占位符
        ]
    )

    # 创建数学代理
    agent = create_openai_tools_agent(llm, tools, prompt)

    # 创建代理执行器
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # 创建消息历史记录对象
    message_history = ChatMessageHistory()
    
    # 给智能体（Agent）添加"对话历史记忆"能力
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        lambda session_id: message_history,
        input_messages_key="input",
        history_messages_key='chat_history',
    )
    return agent_with_chat_history

def create_router_agent():
    """
    创建路由器代理，用于判断用户输入的问题类型并决定调用哪个代理
    这个代理可以直接调用其他代理，而不仅仅是判断
    """
    # 创建聊天模型实例
    llm = ChatOpenAI(
        model="gpt-3.5-turbo:free",
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0
    )

    # 配置路由器工具（可以调用其他代理的工具）
    tools = [invoke_technical_agent, invoke_math_agent]

    # 创建路由器代理的提示模板
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a router assistant that determines which agent should handle the user's request and directly invokes that agent. Use the appropriate tool based on the user's query type. For math-related queries, use invoke_math_agent. For all other queries, use invoke_technical_agent."),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ]
    )

    # 创建路由器代理
    agent = create_openai_tools_agent(llm, tools, prompt)

    # 创建代理执行器
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # 创建消息历史记录对象
    message_history = ChatMessageHistory()
    
    # 给路由器代理添加对话历史记忆能力
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        lambda session_id: message_history,
        input_messages_key="input",
        history_messages_key='chat_history',
    )
    
    return agent_with_chat_history

def main():
    """
    主函数，程序入口点
    """
    # 加载环境变量，从.env文件中读取API密钥等配置
    load_dotenv()
    
    # 获取用户输入
    user_input = input("请输入问题：")
    
    # 创建并运行路由器代理，它会自动判断并调用相应代理
    print("路由器代理正在分析问题并调用相应代理...")
    router_agent = create_router_agent()
    
    result = router_agent.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": "test"}},
    )
    
    print(f"最终回答: {result['output']}")

# 程序入口点
if __name__ == "__main__":
    main()