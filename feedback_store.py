"""
反馈数据存储模块
收集用户对 Agent 回复的评分，作为 DSPy 优化器的训练样本
"""

import json
import os
import time
from typing import Optional

FEEDBACK_FILE = os.path.join(os.path.dirname(__file__), "data", "feedbacks.json")
SKILL_FILE = os.path.join(os.path.dirname(__file__), "data", "skill_state.json")


def _ensure_data_dir():
    os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)


# ============================================================
# 反馈记录
# ============================================================

def save_feedback(
    user_request: str,
    agent_reply: str,
    rating: int,           # 1 = 好评, 0 = 差评
    comment: str = "",
) -> str:
    """保存一条用户反馈，返回反馈 ID"""
    _ensure_data_dir()
    records = load_all_feedbacks()

    fb_id = f"FB{int(time.time() * 1000) % 1_000_000:06d}"
    record = {
        "id": fb_id,
        "user_request": user_request,
        "agent_reply": agent_reply,
        "rating": rating,
        "comment": comment,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    records.append(record)

    with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    return fb_id


def load_all_feedbacks() -> list[dict]:
    """读取所有反馈记录"""
    _ensure_data_dir()
    if not os.path.exists(FEEDBACK_FILE):
        return []
    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def get_positive_feedbacks(min_count: int = 3) -> list[dict]:
    """获取好评记录（用于 Few-Shot 样本）"""
    all_fb = load_all_feedbacks()
    positives = [fb for fb in all_fb if fb["rating"] == 1]
    return positives


def get_stats() -> dict:
    """统计反馈概览"""
    all_fb = load_all_feedbacks()
    total = len(all_fb)
    good = sum(1 for fb in all_fb if fb["rating"] == 1)
    bad = total - good
    return {
        "total": total,
        "good": good,
        "bad": bad,
        "good_rate": round(good / total * 100, 1) if total > 0 else 0,
        "enough_to_optimize": good >= 3,  # 至少需要 3 条好评才能优化
    }


# ============================================================
# Skill 状态（保存优化后的 few-shot 示例和 prompt）
# ============================================================

DEFAULT_SKILL = {
    "version": 0,
    "optimized_at": None,
    "description": "默认 Skill（未经优化）",
    "system_prompt": """你是一位专业友好的电商客服。请根据用户的请求，调用合适的工具来帮助他们解决问题。

可用工具包括：查询用户信息、搜索商品、查询订单、申请退款、提交工单、搜索常见问题。

回复要求：
1. 语气友好、专业
2. 如果需要用户名或订单号等信息，请礼貌地询问
3. 给出清晰明确的答案
4. 如果无法解决，建议用户提交工单""",
    "few_shot_examples": [],  # DSPy 优化后会填入示范对话
}


def load_skill_state() -> dict:
    """读取当前 Skill 状态"""
    _ensure_data_dir()
    if not os.path.exists(SKILL_FILE):
        return DEFAULT_SKILL.copy()
    with open(SKILL_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_skill_state(skill: dict):
    """保存 Skill 状态"""
    _ensure_data_dir()
    with open(SKILL_FILE, "w", encoding="utf-8") as f:
        json.dump(skill, f, ensure_ascii=False, indent=2)
