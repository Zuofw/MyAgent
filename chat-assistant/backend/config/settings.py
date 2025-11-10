import os
from typing import List, Optional

from pydantic import Field, BaseModel


class ModelSettings(BaseModel):
    LLM_MODEL: str = Field("gpt-3.5-turbo:free", description="默认选用的LLM模型")
    EMBEDDING_MODEL: str = Field("text-embedding-ada-002", description="默认选用的Embedding模型")
    VS_TYPE: str = Field("faiss", description="默认选用的向量库类型")
    OPENAI_API_KEY: str = Field("", description="OpenAI API Key")
    OPENAI_BASE_URL: str = Field("", description="OpenAI API Base URL")
    
    class Config:
        env_file = ".env"