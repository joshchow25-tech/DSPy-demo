# DSPy 智能客服 Agent Demo

基于 [DSPy](https://github.com/stanfordnlp/dspy) 框架构建的电商智能客服系统，使用 ReAct 推理模式实现多轮对话和工具调用。

## ✨ 特性

- 🧠 **DSPy ReAct Agent** — 基于 Stanford NLP 的 DSPy 框架，声明式定义 AI Agent
- 🔧 **7 个内置工具** — 用户查询、商品搜索、订单查询、退款、工单、FAQ 等
- 💬 **友好前端界面** — 响应式聊天 UI，快捷操作按钮
- 🛡️ **降级模式** — 无需 LLM API Key 也能运行（基于关键词匹配）
- 🚀 **FastAPI 后端** — RESTful API + 静态页面，一键启动

## 📁 项目结构

```
DSPy-demo/
├── agent.py              # 核心：DSPy Agent（数据模型、工具、ReAct 模块）
├── server.py             # FastAPI 后端服务
├── main.py               # 命令行交互入口（调试用）
├── requirements.txt      # Python 依赖
├── .env.example          # 环境变量模板
├── .gitignore
├── static/
│   └── index.html        # 前端聊天页面
└── README.md
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip3 install -r requirements.txt
```

### 2. 配置环境变量（可选）

```bash
cp .env.example .env
# 编辑 .env，填入你的 API Key
```

> ⚠️ 如果不配置 API Key，系统会自动进入**降级模式**（基于关键词匹配），不依赖任何 LLM 服务也能正常演示。

### 3. 启动服务

```bash
# 启动 Web 服务（前端 + 后端）
python3 server.py
```

然后打开浏览器访问 **http://localhost:8000**

### 4. 命令行模式（调试）

```bash
python3 main.py
```

## 🧩 Agent 架构

```
用户消息
    ↓
┌─────────────────────────┐
│  dspy.ReAct Agent       │
│  ┌───────────────────┐  │
│  │ Signature         │  │  ← 定义输入/输出格式
│  │ user_request →    │  │
│  │ process_result    │  │
│  └───────────────────┘  │
│  ┌───────────────────┐  │
│  │ Tools (7个)       │  │  ← Agent 可调用的工具
│  │ • get_user_info   │  │
│  │ • search_product  │  │
│  │ • get_order_detail│  │
│  │ • get_user_orders │  │
│  │ • apply_refund    │  │
│  │ • submit_ticket   │  │
│  │ • search_faq      │  │
│  └───────────────────┘  │
│  ReAct Loop (max 5)     │  ← 推理 → 行动 → 观察 → ...
└─────────────────────────┘
    ↓
回复用户
```

## 🔌 API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 前端聊天页面 |
| POST | `/api/chat` | 发送消息并获取回复 |
| GET | `/api/health` | 健康检查 |
| GET | `/api/products` | 商品列表 |
| GET | `/api/faq` | 常见问题列表 |

### 聊天接口示例

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "帮我查一下张三的订单"}'
```

## 📝 演示数据

系统内置了以下模拟数据供体验：

- **3 个用户**：张三（金卡会员）、李四（普通会员）、王五（钻石会员）
- **6 款商品**：蓝牙耳机、智能手表、T恤、跑鞋、保温杯、机械键盘
- **3 个订单**：不同状态的订单（已送达、已发货、待发货）
- **8 条 FAQ**：退换货、配送、支付、会员等常见问题

## 🎯 支持的对话场景

- 问候与引导
- 商品搜索与推荐
- 订单查询（按订单号/用户名）
- 退换货申请
- 退款处理
- 会员权益咨询
- 配送/物流查询
- 常见问题自动匹配
- 复杂工单提交

## 📚 关于 DSPy

[DSPy](https://dspy.ai/) 是 Stanford NLP 实验室开发的开源框架，核心理念是 **"Programming, not Prompting"**（编程而非提示工程）。

- **Signature** — 声明式定义 AI 任务的输入/输出
- **Module** — 可组合的 AI 功能模块（如 ReAct、ChainOfThought）
- **Optimizer** — 自动优化提示策略，提升性能

## License

MIT
