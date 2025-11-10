import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import chat

app = FastAPI(title="Chat Assistant API", version="1.0.0")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 包含路由
app.include_router(chat.router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "Welcome to Chat Assistant API"}