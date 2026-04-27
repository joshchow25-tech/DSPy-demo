"""
FastAPI 后端服务
提供 REST API 和前端静态页面
"""

import os
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

from agent import CustomerServiceAgent
from feedback_store import save_feedback, get_stats, load_all_feedbacks, load_skill_state
from skill_optimizer import run_optimization, inject_demo_skill, restore_skill_to_agent

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

# 服务启动时恢复已保存的 Skill
if not agent._fallback_mode:
    restore_skill_to_agent(agent)


# ============================================================
# 请求/响应模型
# ============================================================

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str
    success: bool = True

class FeedbackRequest(BaseModel):
    user_request: str     # 用户的原始问题
    agent_reply: str      # Agent 的回复
    rating: int           # 1=好评, 0=差评
    comment: str = ""


# ============================================================
# 聊天 API
# ============================================================

@app.get("/api/health")
async def health_check():
    """健康检查"""
    skill = load_skill_state()
    return {
        "status": "ok",
        "llm": LM_MODEL,
        "debug": DEBUG_MODE,
        "skill_version": skill.get("version", 0),
        "skill_desc": skill.get("description", "默认 Skill"),
    }

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


# ============================================================
# 反馈 & Skill 优化 API
# ============================================================

@app.post("/api/feedback")
async def submit_feedback(req: FeedbackRequest):
    """
    提交对 Agent 回复的评分（👍/👎）
    - 好评会作为训练样本积累
    - 积累 3 条以上好评后可触发 Skill 优化
    """
    if req.rating not in (0, 1):
        raise HTTPException(status_code=400, detail="rating 只能是 0（差评）或 1（好评）")

    fb_id = save_feedback(
        user_request=req.user_request,
        agent_reply=req.agent_reply,
        rating=req.rating,
        comment=req.comment,
    )
    stats = get_stats()
    return {
        "success": True,
        "feedback_id": fb_id,
        "message": "✅ 反馈已记录，感谢您的评价！",
        "stats": stats,
        "can_optimize": stats["enough_to_optimize"],
        "tip": "已有足够好评，可前往 /api/skill/optimize 触发优化！" if stats["enough_to_optimize"] else f"再积累 {3 - stats['good']} 条好评即可触发 Skill 优化",
    }


@app.get("/api/feedback/stats")
async def feedback_stats():
    """获取反馈统计"""
    stats = get_stats()
    recent = load_all_feedbacks()[-10:]  # 最近 10 条
    return {"stats": stats, "recent": recent}


@app.get("/api/skill")
async def get_skill():
    """获取当前 Skill 状态（版本、few-shot 示例、优化时间等）"""
    skill = load_skill_state()
    return skill


@app.post("/api/skill/optimize")
async def optimize_skill(background_tasks: BackgroundTasks):
    """
    触发 DSPy BootstrapFewShot 自动优化 Skill。
    - 从好评样本中提炼 few-shot 示例
    - 注入 Agent 的 Signature，提升回复质量
    - 版本号自动 +1，可在 /api/skill 查看结果
    """
    stats = get_stats()
    if not stats["enough_to_optimize"]:
        return {
            "success": False,
            "message": f"好评样本不足（当前 {stats['good']} 条），至少需要 3 条才能优化",
            "current_good": stats["good"],
        }

    # 后台异步执行优化（避免阻塞请求）
    # def do_optimize():
    run_optimization(agent, verbose=True)

    # backgroundpy_tasks.add_task(do_optimize)
    return {
        "success": True,
        "message": "⚙️ Skill 优化已在后台启动，通常需要 1-2 分钟，完成后可查看 /api/skill",
        "training_samples": stats["good"],
    }


@app.post("/api/skill/demo")
async def inject_demo():
    """
    注入演示用的 few-shot 示例（无需积累反馈，即时体验 Skill 更新效果）
    注入后 Agent 会参考这些示例来生成更规范的回复格式。
    """
    result = inject_demo_skill(agent)
    return result


@app.post("/api/skill/reset")
async def reset_skill():
    """重置 Skill 到默认状态（清除所有 few-shot 示例）"""
    from feedback_store import DEFAULT_SKILL, save_skill_state
    save_skill_state(DEFAULT_SKILL.copy())
    # 清除 Agent 的 demos
    try:
        for _, predictor in agent._agent.named_predictors():
            predictor.demos = []
    except Exception:
        pass
    return {"success": True, "message": "✅ Skill 已重置到默认版本（无 few-shot 示例）"}


# ============================================================
# 其他 API
# ============================================================

@app.get("/api/products")
async def list_products():
    from agent import product_database
    return [{"id": p.product_id, "name": p.name, "price": p.price, "stock": p.stock, "category": p.category}
            for p in product_database.values()]

@app.get("/api/faq")
async def list_faq():
    from agent import faq_database
    return [{"question": f.question, "answer": f.answer, "category": f.category} for f in faq_database]


# ============================================================
# 前端页面
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse("<h1>前端页面未找到</h1>")

@app.get("/skill", response_class=HTMLResponse)
async def skill_page():
    html_path = os.path.join(os.path.dirname(__file__), "static", "skill.html")
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse("<h1>Skill 管理页未找到</h1>")


# ============================================================
# 启动配置
# ============================================================

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    print(f"🚀 DSPy 智能客服 Agent 启动中...")
    print(f"   LLM:   {LM_MODEL}")
    print(f"   地址:  http://localhost:{port}")
    print(f"   Skill: http://localhost:{port}/skill")
    uvicorn.run(app, host=host, port=port)
