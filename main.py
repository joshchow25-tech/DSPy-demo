"""
智能客服 Agent Demo — 后端入口
基于 DSPy 框架（https://github.com/stanfordnlp/dspy）
启动方式: python3 main.py
"""

from agent import CustomerServiceAgent

def main():
    """命令行交互模式（用于调试）"""
    agent = CustomerServiceAgent()
    print("=" * 50)
    print("🤖 DSPy 智能客服 Agent（命令行模式）")
    print("输入 'quit' 退出")
    print("=" * 50)

    while True:
        user_input = input("\n👤 你: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("再见！👋")
            break

        result = agent.chat(user_input)
        print(f"\n🤖 客服: {result}")


if __name__ == "__main__":
    main()
