"""
DSPy Skill 自动优化模块
======================
核心思路：
1. 收集用户点赞的对话（好评样本）
2. 用 dspy.BootstrapFewShot 从好评样本中自动提炼 few-shot 示例
3. 将优化后的 few-shot 注入 Agent 的 Signature，提升回复质量
4. 保存优化结果（可随时回滚到上一版本）

这就是 DSPy 的核心价值：
  "Programming, not prompting" —— 用数据驱动自动优化提示策略
"""

import dspy
from dspy.teleprompt import BootstrapFewShot
from feedback_store import (
    get_positive_feedbacks,
    load_skill_state,
    save_skill_state,
    get_stats,
)
import time


# ============================================================
# 1. 定义 Metric（评分函数）
#    DSPy 优化器需要一个 metric 来判断哪些示例是"好"的
# ============================================================

def customer_service_metric(example, prediction, trace=None) -> bool:
    """
    评判 Agent 回复质量的 metric。
    - example: 训练样本（含 user_request 和 expected_reply）
    - prediction: Agent 实际生成的 process_result
    返回 True 表示这次回复质量好，可以作为 few-shot 示例
    """
    reply: str = prediction.process_result if hasattr(prediction, "process_result") else str(prediction)

    # 基础质量检查
    if len(reply) < 10:
        return False  # 回复太短

    if any(bad in reply for bad in ["抱歉，处理您的请求时出现了错误", "错误", "exception"]):
        return False  # 包含错误信息

    # 好评样本的标准：用户已经点了赞
    # 如果 example 有 rating 字段，直接用
    if hasattr(example, "rating"):
        return example.rating == 1

    # 默认：回复长度合理就算通过
    return len(reply) >= 20


# ============================================================
# 2. 主优化函数
# ============================================================

def run_optimization(agent_instance, verbose: bool = True) -> dict:
    """
    使用 DSPy BootstrapFewShot 自动优化 Agent 的 Skill。

    参数:
        agent_instance: CustomerServiceAgent 实例
        verbose: 是否打印优化过程

    返回:
        优化结果报告 dict
    """
    stats = get_stats()
    positives = get_positive_feedbacks()

    if verbose:
        print("\n" + "=" * 60)
        print("🚀 DSPy Skill 自动优化开始")
        print(f"   好评样本数量: {len(positives)}")
        print(f"   总体好评率:   {stats['good_rate']}%")
        print("=" * 60)

    if len(positives) < 3:
        msg = f"好评样本不足（当前 {len(positives)} 条，至少需要 3 条），优化跳过"
        if verbose:
            print(f"\n⚠️  {msg}")
        return {"success": False, "message": msg, "samples": len(positives)}

    # ---- 构造 DSPy 训练集 ----
    # 把好评对话转成 dspy.Example 格式
    train_examples = []
    for fb in positives:
        ex = dspy.Example(
            user_request=fb["user_request"],
            process_result=fb["agent_reply"],
        ).with_inputs("user_request")
        train_examples.append(ex)

    if verbose:
        print(f"\n📚 构建训练集: {len(train_examples)} 条样本")
        for i, ex in enumerate(train_examples[:3]):
            print(f"   [{i+1}] Q: {ex.user_request[:40]}...")
            print(f"        A: {ex.process_result[:60]}...")

    # ---- 运行 BootstrapFewShot 优化器 ----
    # BootstrapFewShot 会从训练集中选出最优的 few-shot 示例
    # 注入到 Agent 的 Signature prompt 里
    if verbose:
        print(f"\n⚙️  运行 BootstrapFewShot 优化器...")

    try:
        # 编译前必须清除已有的 demos，否则 BootstrapFewShot 会报错：
        # "AssertionError: Student must be uncompiled."
        for _, predictor in agent_instance._agent.named_predictors():
            if hasattr(predictor, "demos"):
                predictor.demos = []

        if verbose:
            print(f"   已清除已有 demos，准备编译...")

        optimizer = BootstrapFewShot(
            metric=customer_service_metric,
            max_bootstrapped_demos=3,   # 最多保留 3 个示例
            max_labeled_demos=5,        # 最多参考 5 个标注样本
        )

        # 编译优化（核心步骤）
        # 这里会让 Agent 在训练集上跑一遍，筛选出最优示例
        optimized_agent = optimizer.compile(
            student=agent_instance._agent,
            trainset=train_examples,
        )

        # 把优化后的 agent 换回去
        agent_instance._agent = optimized_agent

        # ---- 提取 few-shot 示例并保存到 Skill 状态 ----
        few_shot_demos = _extract_few_shot_demos(optimized_agent)

        # 加载当前 skill 状态并更新
        current_skill = load_skill_state()
        new_version = current_skill.get("version", 0) + 1

        new_skill = {
            "version": new_version,
            "optimized_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "description": f"v{new_version} — 基于 {len(positives)} 条好评样本优化",
            "system_prompt": current_skill.get("system_prompt", ""),
            "few_shot_examples": few_shot_demos,
            "training_samples": len(positives),
        }
        save_skill_state(new_skill)

        result = {
            "success": True,
            "message": f"✅ Skill 已优化到 v{new_version}",
            "version": new_version,
            "few_shot_count": len(few_shot_demos),
            "training_samples": len(positives),
        }

        if verbose:
            print(f"\n✅ 优化完成！")
            print(f"   Skill 版本: v{new_version}")
            print(f"   Few-shot 示例数: {len(few_shot_demos)}")
            print(f"   保存位置: data/skill_state.json")

        return result

    except Exception as e:
        error_msg = f"优化失败: {type(e).__name__}: {str(e)}"
        if verbose:
            print(f"\n❌ {error_msg}")
        return {"success": False, "message": error_msg}


def _extract_few_shot_demos(optimized_agent) -> list[dict]:
    """从优化后的 Agent 中提取 few-shot 示例"""
    demos = []
    try:
        # DSPy 优化后的 demos 存在 predict 模块里
        for name, module in optimized_agent.named_predictors():
            if hasattr(module, "demos") and module.demos:
                for demo in module.demos:
                    if hasattr(demo, "user_request") and hasattr(demo, "process_result"):
                        demos.append({
                            "user_request": demo.user_request,
                            "process_result": demo.process_result,
                        })
                break  # 只取第一个 predictor 的 demos
    except Exception:
        pass
    return demos


# ============================================================
# 3. 加载已保存的 Skill（服务重启后恢复优化状态）
# ============================================================

def restore_skill_to_agent(agent_instance) -> bool:
    """
    从持久化存储中恢复已优化的 Skill 到 Agent 实例。
    服务重启时调用，保证优化成果不丢失。
    """
    skill = load_skill_state()
    if not skill.get("few_shot_examples"):
        return False  # 还没有优化过

    try:
        demos = []
        for ex in skill["few_shot_examples"]:
            demo = dspy.Example(
                user_request=ex["user_request"],
                process_result=ex["process_result"],
            ).with_inputs("user_request")
            demos.append(demo)

        # 将 few-shot 注入到 ReAct 的最后一个 predictor（汇总输出步骤）
        predictors = list(agent_instance._agent.named_predictors())
        if predictors:
            _, last_predictor = predictors[-1]
            last_predictor.demos = demos

        print(f"✅ 已恢复 Skill v{skill['version']}（{len(demos)} 个 few-shot 示例）")
        return True
    except Exception as e:
        print(f"⚠️  Skill 恢复失败: {e}")
        return False


# ============================================================
# 4. 演示：快速注入示例（不需要跑优化器，直接手动设置）
# ============================================================

def inject_demo_skill(agent_instance) -> dict:
    """
    演示模式：直接注入预设的 few-shot 示例，不需要收集真实反馈。
    用于展示「Skill 更新后 Agent 回复风格变化」的效果。
    """
    demo_examples = [
        dspy.Example(
            user_request="我想退货",
            process_result="""您好！非常抱歉给您带来了不便 😊

**退货流程如下：**
1. 收到商品 **7天内** 可申请无理由退货
2. 请提供您的订单号，我来帮您发起申请
3. 审核通过后 **1-3天** 完成退款

请问您的订单号是多少？我立刻为您处理 💪""",
        ).with_inputs("user_request"),

        dspy.Example(
            user_request="快递什么时候到？",
            process_result="""您好！关于配送时效说明如下 🚚

- **普通快递**：下单后 1-3天 发货，全国 3-7天 到达
- **顺丰加急**：次日达（部分地区，额外收费）
- 发货后会通过 **短信 + APP** 推送物流信息

如需查询具体订单的物流状态，请告诉我您的订单号 📦""",
        ).with_inputs("user_request"),

        dspy.Example(
            user_request="有优惠券吗？",
            process_result="""当然有！🎉 以下是我们目前的优惠方式：

| 方式 | 优惠力度 |
|------|---------|
| 🥈 银卡会员 | 95折 |
| 🥇 金卡会员 | 9折 |
| 💎 钻石会员 | 85折 |
| 新人专享 | 满100减15 |

您可以在「领券中心」领取最新优惠券。请问您目前是什么会员等级？""",
        ).with_inputs("user_request"),
    ]

    try:
        predictors = list(agent_instance._agent.named_predictors())
        if predictors:
            _, last_predictor = predictors[-1]
            last_predictor.demos = demo_examples

        # 保存到 skill state
        skill = load_skill_state()
        skill["version"] = skill.get("version", 0) + 1
        skill["optimized_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        skill["description"] = f"v{skill['version']} — 演示模式注入（3条精选示例）"
        skill["few_shot_examples"] = [
            {"user_request": ex.user_request, "process_result": ex.process_result}
            for ex in demo_examples
        ]
        save_skill_state(skill)

        return {
            "success": True,
            "version": skill["version"],
            "message": f"演示 Skill v{skill['version']} 已注入（3 个 few-shot 示例）",
            "examples": [ex.user_request for ex in demo_examples],
        }
    except Exception as e:
        return {"success": False, "message": str(e)}
