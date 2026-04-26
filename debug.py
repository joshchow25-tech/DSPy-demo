#!/usr/bin/env python3
"""
DSPy ReAct 调试工具
用法:
    python3 debug.py                    # 交互式调试
    python3 debug.py "查一下张三的订单"   # 单条调试
    DSPY_DEBUG=true python3 debug.py     # 开启详细 trace
"""

import sys
import os
from dotenv import load_dotenv

load_dotenv()


def main():
    from agent import CustomerServiceAgent

    lm_model = os.getenv("DSPY_LM", "deepseek/deepseek-chat")
    debug = os.getenv("DSPY_DEBUG", "true").lower() in ("true", "1", "yes")

    print("=" * 60)
    print("🔧 DSPy ReAct 调试模式")
    print(f"   LLM:    {lm_model}")
    print(f"   Debug:  {'✅ 开启 — 打印完整推理轨迹' if debug else '❌ 关闭'}")
    print(f"   提示:   设置 DSPY_DEBUG=true 开启详细 trace")
    print("=" * 60)

    agent = CustomerServiceAgent(lm_model=lm_model, debug=debug)

    # 单条模式
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"\n👤 用户: {query}\n")
        reply = agent.chat(query)
        print(f"\n🤖 客服: {reply}\n")
        # 调用 inspect_history 查看完整 LLM 交互
        if debug:
            print("\n📚 最近 1 次 LLM 调用详情:")
            print("-" * 40)
            agent.inspect_history(n=1)
        return

    # 交互模式
    # print("\n输入消息开始调试，输入 'history' 查看 LLM 调用历史，输入 'quit' 退出\n")
    # while True:
    #     try:
    #         user_input = input("👤 你: ").strip()
    #     except (EOFError, KeyboardInterrupt):
    #         print("\n再见！")
    #         break

    #     if not user_input:
    #         continue
    #     if user_input.lower() in ("quit", "exit", "q"):
    #         print("再见！")
    #         break
    #     if user_input.lower() == "history":
    #         print("\n📚 最近 3 次 LLM 调用详情:")
    #         print("-" * 40)
    #         agent.inspect_history(n=3)
    #         continue
    #     if user_input.lower() == "debug":
    #         agent.debug = not agent.debug
    #         print(f"\n🔄 Debug 模式: {'✅ 开启' if agent.debug else '❌ 关闭'}\n")
    #         continue

    #     print()  # 空行分隔
    #     reply = agent.chat(user_input)
    #     print(f"\n🤖 客服: {reply}\n")


if __name__ == "__main__":
    main()
