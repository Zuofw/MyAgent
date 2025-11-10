from pydantic import BaseModel
from typing import List, Dict, Optional, Union


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    stream: Optional[bool] = False


class ChatResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Dict]
    usage: Optional[Dict] = None


class StreamChatResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Dict]