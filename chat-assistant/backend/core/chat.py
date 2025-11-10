from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
import os


class ChatCore:
    def __init__(self, model_name: str = "gpt-3.5-turbo:free", base_url: str = None, api_key: str = None):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
            
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. Answer the user's questions clearly and accurately."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])
        
        self.chat_history = ChatMessageHistory()
    
    def _get_model(self):
        # 如果没有提供base_url和api_key，则从环境变量中获取
        base_url = self.base_url or os.getenv("OPENAI_BASE_URL")
        api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        
        # 检查是否提供了API密钥
        if not api_key:
            raise ValueError("API key is required. Please set OPENAI_API_KEY in your .env file.")
        
        return ChatOpenAI(
            model=self.model_name,
            base_url=base_url,
            api_key=api_key,
            temperature=0.7
        )
        
    def _get_chain(self):
        try:
            model = self._get_model()
        except ValueError as e:
            raise ValueError(f"Error initializing model: {str(e)}")
        
        chain = self.prompt | model
        chain_with_history = RunnableWithMessageHistory(
            chain,
            lambda session_id: self.chat_history,
            input_messages_key="input",
            history_messages_key="history"
        )
        return chain_with_history
    
    def chat(self, message: str, session_id: str = "default") -> str:
        try:
            chain_with_history = self._get_chain()
            response = chain_with_history.invoke(
                {"input": message},
                config={"configurable": {"session_id": session_id}}
            )
            return response.content
        except Exception as e:
            raise Exception(f"Error during chat: {str(e)}")