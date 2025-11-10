from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage, AIMessage

from api.models.chat import ChatRequest, ChatResponse, StreamChatResponse
from core.chat import ChatCore
import uuid
import time
import logging

router = APIRouter()

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化聊天核心
try:
    chat_core = ChatCore()
    logger.info("ChatCore initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize ChatCore: {e}")
    chat_core = None

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if chat_core is None:
        raise HTTPException(status_code=500, detail="ChatCore is not initialized")
    
    try:
        # 获取最新的用户消息
        user_message = request.messages[-1].content if request.messages else ""
        logger.info(f"Received message: {user_message}")
        
        # 调用聊天核心处理
        response_text = chat_core.chat(user_message)
        logger.info(f"Generated response: {response_text}")
        
        # 构造响应
        response = ChatResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:10]}",
            object="chat.completion",
            created=int(time.time()),
            model="gpt-3.5-turbo",
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }]
        )
        
        return response
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))