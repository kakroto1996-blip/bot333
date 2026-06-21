"""
WhatsApp AI Chatbot - FastAPI + LangGraph + OpenRouter
سريع، موثوق، ويدعم العربية بشكل كامل
"""

import os
import json
import logging
import sqlite3
import requests
from contextlib import asynccontextmanager
from typing import Annotated, TypedDict
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import PlainTextResponse, JSONResponse
from openai import OpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────
PHONE_NUMBER_ID  = os.environ["WHATSAPP_PHONE_NUMBER_ID"]
ACCESS_TOKEN     = os.environ["WHATSAPP_ACCESS_TOKEN"]
VERIFY_TOKEN     = os.environ["WHATSAPP_VERIFY_TOKEN"]
OPENROUTER_KEY   = os.environ["OPENROUTER_API_KEY"]
MODEL            = os.environ.get("AI_MODEL", "openai/o4-mini")
DB_PATH          = os.environ.get("DB_PATH", "sessions.db")

WA_URL = f"https://graph.facebook.com/v21.0/{PHONE_NUMBER_ID}/messages"

SYSTEM_PROMPT = SYSTEM_PROMPT = """الدور: أنت مساعد ذكي ومحترف لخدمة العملاء.
سياسة اللغة: تواصل باللغة العربية الفصحى فقط.

مهمتك: جمع 3 معلومات خطوة بخطوة (معلومة واحدة في كل رسالة):
1. الاسم الثلاثي
2. رقم الهوية
3. رقم الجوال

قواعد صارمة (يجب اتباعها بدقة):
- لا تطلب أكثر من معلومة واحدة في كل رد.
- إذا قام المستخدم بتزويدك بمعلومة، لا تطلبها مجدداً، وانتقل فوراً للخطوة التالية.
- لا تكرر السؤال السابق إذا أجاب المستخدم عليه بالفعل.
- إذا أرسل المستخدم زر (تتبع الطلب أو الدعم الفني)، ابدأ بطلب الاسم الثلاثي فوراً.
- التزم بالهيدوء والاحترافية: إذا أرسل المستخدم رسالة لا تحتوي على المعلومة المطلوبة، اطلبها منه بأسلوب مهذب مرة واحدة فقط.
- ممنوع منعاً باتاً إرسال أكثر من رسالة واحدة في كل رد من طرفك.

بعد جمع المعلومات الثلاث أرسل هذه الرسالة حرفياً:
شكراً لك يا [الاسم الثلاثي]. صاحب الهوية ([رقم الهوية]) لقد تم استلام بياناتك بنجاح. نحن نقدر تعاونك معنا. سوف يتم تحويلك للموظف بأسرع وقت.
أضف في نهاية ردك الأخير حصراً: [DONE]"""
# ─── OpenRouter client ────────────────────────────────────────────────────────
ai = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_KEY,
)

# ─── SQLite Sessions ──────────────────────────────────────────────────────────
def init_db() -> None:
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            phone TEXT PRIMARY KEY,
            history TEXT NOT NULL DEFAULT '[]',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    con.commit()
    con.close()
    log.info("Database initialized at %s", DB_PATH)

def load_session(phone: str) -> list[dict]:
    con = sqlite3.connect(DB_PATH)
    row = con.execute("SELECT history FROM sessions WHERE phone=?", (phone,)).fetchone()
    con.close()
    return json.loads(row[0]) if row else []

def save_session(phone: str, history: list[dict]) -> None:
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT INTO sessions (phone, history) VALUES (?,?) "
        "ON CONFLICT(phone) DO UPDATE SET history=excluded.history",
        (phone, json.dumps(history, ensure_ascii=False)),
    )
    con.commit()
    con.close()

def delete_session(phone: str) -> None:
    con = sqlite3.connect(DB_PATH)
    con.execute("DELETE FROM sessions WHERE phone=?", (phone,))
    con.commit()
    con.close()
    log.info("Session deleted for %s", phone)

# ─── WhatsApp API ─────────────────────────────────────────────────────────────
def _wa_headers() -> dict:
    return {"Authorization": f"Bearer {ACCESS_TOKEN}", "Content-Type": "application/json"}

def wa_send_text(to: str, text: str) -> None:
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": text},
    }
    r = requests.post(WA_URL, json=payload, headers=_wa_headers(), timeout=10)
    if not r.ok:
        log.error("WhatsApp error: %s", r.text)

def wa_send_buttons(to: str) -> None:
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "interactive",
        "interactive": {
            "type": "button",
            "body": {"text": "مرحباً بك! 👋\nكيف يمكنني مساعدتك اليوم؟"},
            "action": {
                "buttons": [
                    {"type": "reply", "reply": {"id": "order_tracking",    "title": "📦 تتبع الطلب"}},
                    {"type": "reply", "reply": {"id": "technical_support", "title": "🛠 الدعم الفني"}},
                ]
            },
        },
    }
    r = requests.post(WA_URL, json=payload, headers=_wa_headers(), timeout=10)
    if not r.ok:
        log.error("WhatsApp buttons error: %s", r.text)

# ─── LangGraph State ──────────────────────────────────────────────────────────
class BotState(TypedDict):
    messages: Annotated[list, add_messages]
    phone: str
    done: bool

def ai_node(state: BotState) -> BotState:
    # 1. إعداد الـ history كما كنت تفعل
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            history.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            history.append({"role": "assistant", "content": msg.content})

    # 2. استدعاء النموذج
    response = ai.chat.completions.create(model=MODEL, messages=history)
    reply = response.choices[0].message.content or ""

    done = "[DONE]" in reply
    clean = reply.replace("[DONE]", "").strip()

    # 3. الحل: إرسال الرسالة فقط إذا كان الـ state لا يحتوي على هذا الرد مسبقاً
    # أو نكتفي بإرسالها هنا، ولكن نضبط الـ conditional edges لضمان عدم التكرار
    wa_send_text(state["phone"], clean)
    
    return {"messages": [AIMessage(content=reply)], "done": done}

def should_end(state: BotState) -> str:
    return END if state.get("done") else "ai"

# بناء الجراف
builder = StateGraph(BotState)
builder.add_node("ai", ai_node)
builder.set_entry_point("ai")
builder.add_conditional_edges("ai", should_end, {"ai": "ai", END: END})
graph = builder.compile()

# ─── Core Logic ───────────────────────────────────────────────────────────────
def handle_message(phone: str, user_input: str) -> None:
    history = load_session(phone)

    if not history:
        # 1. إرسال الأزرار
        wa_send_buttons(phone)
        
        # 2. حفظ رسالة الترحيب في الـ history فوراً لمنع تكرارها
        welcome_msg = "مرحباً بك! 👋 كيف يمكنني مساعدتك اليوم؟"
        new_history = [{"role": "assistant", "content": welcome_msg}]
        save_session(phone, new_history)
        
        log.info("New user %s - buttons sent and session initialized", phone)
        return
    
    # ... باقي الكود كما هو

    # تحويل history إلى LangGraph messages
    messages = []
    for item in history:
        if item["role"] == "user":
            messages.append(HumanMessage(content=item["content"]))
        else:
            messages.append(AIMessage(content=item["content"]))
    messages.append(HumanMessage(content=user_input))

    state = graph.invoke({"messages": messages, "phone": phone, "done": False})

    if state.get("done"):
        delete_session(phone)
    else:
        # حفظ المحادثة المحدّثة
        new_history = history + [
            {"role": "user",      "content": user_input},
            {"role": "assistant", "content": state["messages"][-1].content},
        ]
        save_session(phone, new_history)

# ─── FastAPI App ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    log.info("Bot started | Model: %s", MODEL)
    yield

app = FastAPI(title="WhatsApp AI Bot", lifespan=lifespan)


@app.get("/webhook")
def verify(request: Request):
    """التحقق من الويب هوك مع Meta."""
    params = request.query_params
    if params.get("hub.mode") == "subscribe" and params.get("hub.verify_token") == VERIFY_TOKEN:
        log.info("Webhook verified ✓")
        return PlainTextResponse(params.get("hub.challenge", ""))
    return PlainTextResponse("Forbidden", status_code=403)


@app.post("/webhook")
async def webhook(request: Request, background: BackgroundTasks):
    """استقبال رسائل واتساب - يرد 200 فوراً ثم يعالج في الخلفية."""
    data = await request.json()

    if data.get("object") != "whatsapp_business_account":
        return JSONResponse({"status": "ignored"})

    try:
        value    = data["entry"][0]["changes"][0]["value"]
        messages = value.get("messages")
        if not messages:
            return JSONResponse({"status": "no_messages"})

        message     = messages[0]
        from_number = message["from"]
        display     = value["metadata"]["display_phone_number"]

        # تجاهل رسائل النظام والبوت نفسه
        if from_number == display:
            return JSONResponse({"status": "self"})

        msg_type = message.get("type")
        if msg_type == "text":
            user_input = message["text"]["body"].strip()
        elif msg_type == "interactive":
            user_input = message["interactive"]["button_reply"]["title"]
        else:
            return JSONResponse({"status": "unsupported"})

        # ← الرد على WhatsApp فوراً بـ 200، ثم المعالجة في الخلفية
        background.add_task(handle_message, from_number, user_input)

    except (KeyError, IndexError) as e:
        log.warning("Parse error: %s", e)

    return JSONResponse({"status": "ok"})


@app.get("/health")
def health():
    return {"status": "running", "model": MODEL}
