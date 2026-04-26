"""
客服 Agent 核心模块
基于 DSPy ReAct 模式，支持工具调用和多轮推理
"""

import random
import string
from pydantic import BaseModel
from typing import Optional


# ============================================================
# 1. 数据模型定义
# ============================================================

class Date(BaseModel):
    """日期时间"""
    year: int
    month: int
    day: int
    hour: int = 0


class UserProfile(BaseModel):
    """用户信息"""
    user_id: str
    name: str
    email: str
    phone: str = ""
    vip_level: int = 0  # 0=普通, 1=银卡, 2=金卡, 3=钻石


class Product(BaseModel):
    """商品信息"""
    product_id: str
    name: str
    price: float
    stock: int
    category: str


class Order(BaseModel):
    """订单信息"""
    order_id: str
    user_profile: UserProfile
    product: Product
    quantity: int
    total_price: float
    status: str  # pending, shipped, delivered, cancelled, refunding
    create_time: str


class Ticket(BaseModel):
    """客服工单"""
    ticket_id: str
    user_profile: UserProfile
    issue: str
    status: str = "open"  # open, processing, resolved, closed


class FAQ(BaseModel):
    """常见问题"""
    question: str
    answer: str
    category: str


# ============================================================
# 2. 模拟数据库
# ============================================================

user_database = {
    "张三": UserProfile(user_id="U001", name="张三", email="zhangsan@example.com", phone="13800138001", vip_level=2),
    "李四": UserProfile(user_id="U002", name="李四", email="lisi@example.com", phone="13800138002", vip_level=0),
    "王五": UserProfile(user_id="U003", name="王五", email="wangwu@example.com", phone="13800138003", vip_level=3),
}

product_database = {
    "P001": Product(product_id="P001", name="无线蓝牙耳机 Pro", price=299.0, stock=150, category="电子产品"),
    "P002": Product(product_id="P002", name="智能手表 S3", price=1299.0, stock=80, category="电子产品"),
    "P003": Product(product_id="P003", name="纯棉T恤 经典款", price=89.0, stock=500, category="服装"),
    "P004": Product(product_id="P004", name="运动跑鞋 飞翼系列", price=599.0, stock=200, category="鞋类"),
    "P005": Product(product_id="P005", name="保温杯 大容量", price=128.0, stock=300, category="生活用品"),
    "P006": Product(product_id="P006", name="机械键盘 RGB", price=459.0, stock=120, category="电子产品"),
}

order_database = {
    "ORD20250101001": Order(
        order_id="ORD20250101001",
        user_profile=user_database["张三"],
        product=product_database["P001"],
        quantity=2,
        total_price=598.0,
        status="delivered",
        create_time="2025-01-01 10:30:00",
    ),
    "ORD20250115002": Order(
        order_id="ORD20250115002",
        user_profile=user_database["张三"],
        product=product_database["P002"],
        quantity=1,
        total_price=1299.0,
        status="shipped",
        create_time="2025-01-15 14:20:00",
    ),
    "ORD20250201003": Order(
        order_id="ORD20250201003",
        user_profile=user_database["李四"],
        product=product_database["P003"],
        quantity=3,
        total_price=267.0,
        status="pending",
        create_time="2025-02-01 09:15:00",
    ),
}

ticket_database: dict[str, Ticket] = {
    "TK001": Ticket(
        ticket_id="TK001",
        user_profile=user_database["张三"],
        issue="耳机左侧没有声音",
        status="processing",
    ),
}

faq_database = [
    FAQ(question="如何退换货？", answer="在收到商品7天内可申请无理由退换货，请在「我的订单」中点击「申请退换」，填写原因后等待审核。运费由我们承担。", category="售后"),
    FAQ(question="配送需要多长时间？", answer="一般下单后1-3天发货，国内大部分地区3-7天到达。偏远地区可能需要7-15天。支持顺丰加急（额外收费）。", category="物流"),
    FAQ(question="支持哪些支付方式？", answer="支持支付宝、微信支付、银联卡、信用卡等主流支付方式。大额订单还可申请分期付款。", category="支付"),
    FAQ(question="如何查看订单状态？", answer="登录后进入「我的订单」页面，可实时查看订单状态。发货后会通过短信和APP推送通知。", category="订单"),
    FAQ(question="如何修改收货地址？", answer="订单未发货前可在「我的订单」中修改收货地址。已发货的订单请联系客服协助处理。", category="订单"),
    FAQ(question="有会员优惠吗？", answer="我们有银卡、金卡、钻石三级会员制度。银卡享95折，金卡享9折，钻石享85折。消费满额自动升级。", category="会员"),
    FAQ(question="商品保修政策是什么？", answer="电子产品保修一年，人为损坏除外。服装类7天内有质量问题可换新。具体以商品详情页说明为准。", category="售后"),
    FAQ(question="发票怎么开？", answer="下单时可选择是否需要发票。支持电子发票和纸质发票。电子发票下单后自动发送至邮箱，纸质发票随货寄出。", category="支付"),
]


# ============================================================
# 3. 工具函数（Agent 可调用的能力）
# ============================================================

def get_user_info(name: str) -> dict:
    """根据用户名查询用户信息，包括会员等级"""
    user = user_database.get(name)
    if user:
        vip_names = {0: "普通会员", 1: "银卡会员", 2: "金卡会员", 3: "钻石会员"}
        return {
            "user_id": user.user_id,
            "name": user.name,
            "email": user.email,
            "phone": user.phone,
            "vip_level": vip_names.get(user.vip_level, "未知"),
        }
    return {"error": f"未找到用户 '{name}' 的信息"}


def search_product(keyword: str, category: str = "") -> list[dict]:
    """根据关键词和分类搜索商品"""
    results = []
    for p in product_database.values():
        match_name = keyword.lower() in p.name.lower()
        match_cat = category == "" or category.lower() in p.category.lower()
        if match_name or match_cat:
            results.append({
                "product_id": p.product_id,
                "name": p.name,
                "price": p.price,
                "stock": p.stock,
                "category": p.category,
            })
    return results if results else [{"message": f"未找到与 '{keyword}' 相关的商品"}]


def get_order_detail(order_id: str) -> dict:
    """根据订单号查询订单详情"""
    order = order_database.get(order_id)
    if order:
        return {
            "order_id": order.order_id,
            "product": order.product.name,
            "quantity": order.quantity,
            "total_price": order.total_price,
            "status": order.status,
            "status_desc": {"pending": "待发货", "shipped": "已发货", "delivered": "已送达", "cancelled": "已取消", "refunding": "退款中"}.get(order.status, order.status),
            "create_time": order.create_time,
        }
    return {"error": f"未找到订单 '{order_id}'"}


def get_user_orders(user_name: str) -> list[dict]:
    """查询某个用户的所有订单"""
    results = []
    for order in order_database.values():
        if order.user_profile.name == user_name:
            results.append({
                "order_id": order.order_id,
                "product": order.product.name,
                "total_price": order.total_price,
                "status": order.status,
            })
    return results if results else [{"message": f"用户 '{user_name}' 暂无订单记录"}]


def apply_refund(order_id: str, reason: str) -> dict:
    """为订单申请退款"""
    order = order_database.get(order_id)
    if not order:
        return {"error": f"未找到订单 '{order_id}'"}
    if order.status in ("cancelled", "refunding"):
        return {"error": f"订单 '{order_id}' 当前状态为 '{order.status}'，无法重复申请退款"}
    order.status = "refunding"
    return {
        "success": True,
        "message": f"订单 '{order_id}' 退款申请已提交，预计3-5个工作日处理完成",
        "refund_amount": order.total_price,
        "reason": reason,
    }


def submit_ticket(user_name: str, issue: str) -> dict:
    """提交客服工单"""
    user = user_database.get(user_name)
    if not user:
        return {"error": f"未找到用户 '{user_name}'，请先确认用户名"}
    ticket_id = "TK" + ''.join(random.choices(string.digits, k=3))
    ticket = Ticket(ticket_id=ticket_id, user_profile=user, issue=issue)
    ticket_database[ticket_id] = ticket
    return {
        "success": True,
        "ticket_id": ticket_id,
        "message": f"工单 {ticket_id} 已创建，客服将在24小时内与您联系",
        "issue": issue,
    }


def search_faq(question: str) -> list[dict]:
    """在常见问题库中搜索匹配的问题"""
    results = []
    keywords = question.lower().replace("？", "").replace("?", "").split()
    for faq in faq_database:
        score = sum(1 for kw in keywords if kw in faq.question.lower() or kw in faq.answer.lower())
        if score > 0:
            results.append({"question": faq.question, "answer": faq.answer, "category": faq.category, "relevance": score})
    results.sort(key=lambda x: x["relevance"], reverse=True)
    return results[:3] if results else [{"message": "未找到相关问题，建议提交工单由人工客服处理"}]


# ============================================================
# 4. DSPy Agent 类
# ============================================================

import dspy


class CustomerServiceAgent:
    """
    基于 DSPy ReAct 的智能客服 Agent
    - 使用 dspy.Signature 定义输入/输出格式
    - 使用 dspy.ReAct 实现推理-行动循环
    - 支持多工具调用和上下文累积
    """

    def __init__(self, lm_model: str = "openai/gpt-4o-mini"):
        # 配置语言模型
        try:
            dspy.configure(lm=dspy.LM(lm_model))
        except Exception as e:
            print(f"⚠️  LLM 配置失败: {e}")
            print("   将使用降级模式（仅关键词匹配）")
            self._fallback_mode = True
            return

        self._fallback_mode = False

        # 定义 DSPy Signature：输入用户请求，输出处理结果
        class CustomerServiceSignature(dspy.Signature):
            """你是一位专业友好的电商客服。请根据用户的请求，调用合适的工具来帮助他们解决问题。

            可用工具包括：查询用户信息、搜索商品、查询订单、申请退款、提交工单、搜索常见问题。

            回复要求：
            1. 语气友好、专业
            2. 如果需要用户名或订单号等信息，请礼貌地询问
            3. 给出清晰明确的答案
            4. 如果无法解决，建议用户提交工单
            """
            user_request: str = dspy.InputField(desc="用户的请求或问题")
            process_result: str = dspy.OutputField(desc="对用户的回复，包含处理结果和必要的信息")

        # 创建 ReAct Agent，传入 Signature 和工具列表
        self._agent = dspy.ReAct(
            CustomerServiceSignature,
            tools=[
                get_user_info,
                search_product,
                get_order_detail,
                get_user_orders,
                apply_refund,
                submit_ticket,
                search_faq,
            ],
            max_iters=5,  # 最多5轮推理-行动循环
        )

    def chat(self, user_message: str) -> str:
        """处理用户消息并返回回复"""
        if self._fallback_mode:
            return self._fallback_chat(user_message)

        try:
            result = self._agent(user_request=user_message)
            return result.process_result
        except Exception as e:
            # 如果 Agent 出错，尝试降级模式
            print(f"⚠️  Agent 执行出错: {e}")
            return self._fallback_chat(user_message)

    def _fallback_chat(self, user_message: str) -> str:
        """降级模式：基于关键词的简单匹配（不需要 LLM）"""
        msg = user_message.lower()

        # 问候
        if any(g in msg for g in ["你好", "嗨", "在吗", "hello", "hi"]):
            return "您好！欢迎来到智能客服中心 😊 请问有什么可以帮您的？"

        # 查订单
        if "订单" in msg and any(o in msg for o in order_database):
            for oid in order_database:
                if oid in msg:
                    order = order_database[oid]
                    return f"📋 订单详情：\n- 订单号: {order.order_id}\n- 商品: {order.product.name}\n- 数量: {order.quantity}\n- 金额: ¥{order.total_price}\n- 状态: {order.status}\n\n请问还有什么需要帮助的吗？"

        # 查商品
        if "商品" in msg or "产品" in msg or "有没有" in msg:
            results = []
            for p in product_database.values():
                if any(kw in p.name.lower() for kw in msg.split()):
                    results.append(f"- {p.name} | ¥{p.price} | 库存:{p.stock}")
            if results:
                return "🔍 找到以下商品：\n" + "\n".join(results) + "\n\n请问您需要哪款？"
            return "🔍 抱歉，没有找到匹配的商品。您可以换个关键词试试，或告诉我您想要什么类型的商品？"

        # 退换货
        if "退" in msg or "换" in msg or "退款" in msg:
            return "📦 退换货说明：\n- 收到商品7天内可申请无理由退换货\n- 请在「我的订单」中点击「申请退换」\n- 如需我帮您操作，请提供订单号\n- 运费由我们承担 ✅"

        # 配送
        if "配送" in msg or "快递" in msg or "发货" in msg or "物流" in msg:
            return "🚚 配送说明：\n- 下单后1-3天发货\n- 国内大部分地区3-7天到达\n- 发货后会短信通知\n- 支持顺丰加急"

        # 支付
        if "支付" in msg or "付款" in msg or "发票" in msg:
            return "💳 支付说明：\n- 支持支付宝、微信、银联、信用卡\n- 大额订单可分期\n- 发票：电子发票自动发邮箱，纸质发票随货寄出"

        # 会员
        if "会员" in msg or "vip" in msg:
            return "👑 会员制度：\n- 🥈 银卡会员：享95折\n- 🥇 金卡会员：享9折\n- 💎 钻石会员：享85折\n- 消费满额自动升级\n\n请问您想了解哪个等级的权益？"

        # 默认回复
        return "感谢您的咨询！请更详细地描述您的问题，例如：\n- \"查询我的订单\"\n- \"搜索耳机\"\n- \"如何退款\"\n- \"配送要多久\"\n\n我会尽力为您提供帮助 😊"
