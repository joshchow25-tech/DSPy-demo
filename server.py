"""
FastAPI 后端服务
提供 REST API 和前端静态页面
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

from agent import CustomerServiceAgent

# 加载环境变量
load_dotenv()

app = FastAPI(title="DSPy 智能客服", description="基于 DSPy 框架的智能客服 Agent Demo")

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化 Agent
LM_MODEL = os.getenv("DSPY_LM", "deepseek/deepseek-chat")
DEBUG_MODE = os.getenv("DSPY_DEBUG", "false").lower() in ("true", "1", "yes")
agent = CustomerServiceAgent(lm_model=LM_MODEL, debug=DEBUG_MODE)


# ============================================================
# 请求/响应模型
# ============================================================

class ChatRequest(BaseModel):
    """聊天请求"""
    message: str


class ChatResponse(BaseModel):
    """聊天响应"""
    reply: str
    success: bool = True


# ============================================================
# API 路由
# ============================================================

@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {"status": "ok", "service": "DSPy 智能客服", "llm": LM_MODEL, "debug": DEBUG_MODE}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """处理用户消息"""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="消息不能为空")

    try:
        reply = agent.chat(request.message)
        return ChatResponse(reply=reply, success=True)
    except Exception as e:
        return ChatResponse(reply=f"抱歉，处理您的请求时出现了错误：{str(e)}", success=False)


@app.get("/api/products")
async def list_products():
    """获取商品列表"""
    from agent import product_database
    return [
        {
            "id": p.product_id,
            "name": p.name,
            "price": p.price,
            "stock": p.stock,
            "category": p.category,
        }
        for p in product_database.values()
    ]


@app.get("/api/faq")
async def list_faq():
    """获取常见问题列表"""
    from agent import faq_database
    return [{"question": f.question, "answer": f.answer, "category": f.category} for f in faq_database]


# ============================================================
# 前端页面
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def index():
    """返回前端聊天页面"""
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse("<h1>前端页面未找到，请确保 static/index.html 存在</h1>")


# ============================================================
# 启动配置
# ============================================================

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    print(f"🚀 DSPy 智能客服 Agent 启动中...")
    print(f"   LLM: {LM_MODEL}")
    print(f"   地址: http://localhost:{port}")
    uvicorn.run(app, host=host, port=port)
