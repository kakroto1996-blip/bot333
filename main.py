"""
WhatsApp AI Chatbot - FastAPI + LangGraph + OpenRouter
سريع، موثوق، ويدعم العربية بشكل كامل
"""

import os
import re
import json
import logging
import sqlite3
import smtplib
import threading
import requests
from datetime import datetime
from email.mime.text import MIMEText
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Annotated, TypedDict
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import PlainTextResponse, JSONResponse
from openai import OpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import gspread
from google.oauth2.service_account import Credentials

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
MODEL            = os.environ.get("AI_MODEL", "openai/gpt-4o-mini")
DB_PATH          = os.environ.get("DB_PATH", "sessions.db")

# ─── Google Sheets + Gmail (اختياري: البوت يعمل حتى لو لم تُضبط هذه القيم) ───
GOOGLE_SHEET_ID              = os.environ.get("GOOGLE_SHEET_ID", "")
GOOGLE_SERVICE_ACCOUNT_JSON  = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "")
SMTP_EMAIL                   = os.environ.get("SMTP_EMAIL", "")
SMTP_APP_PASSWORD            = os.environ.get("SMTP_APP_PASSWORD", "")
NOTIFY_EMAIL                 = os.environ.get("NOTIFY_EMAIL", SMTP_EMAIL)

WA_URL = f"https://graph.facebook.com/v21.0/{PHONE_NUMBER_ID}/messages"

SYSTEM_PROMPT = """الدور: أنت مساعد ذكي ومحترف لخدمة العملاء.
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
    con.execute("PRAGMA journal_mode=WAL;")  # يحسّن التزامن عند الكتابة المتعددة
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

# ─── Per-Phone Locks ──────────────────────────────────────────────────────────
# يمنع تضارب معالجة رسالتين متتاليتين من نفس العميل في نفس الوقت
_phone_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
_locks_guard = threading.Lock()  # يحمي القاموس نفسه عند إنشاء قفل جديد

def get_phone_lock(phone: str) -> threading.Lock:
    with _locks_guard:
        return _phone_locks[phone]

# ─── Google Sheets ────────────────────────────────────────────────────────────
_sheets_client = None  # يُهيَّأ مرة واحدة فقط ويُعاد استخدامه (تجنّب إعادة المصادقة كل مرة)

def get_sheet():
    """يرجع أول ورقة (worksheet) في الشيت، أو None إذا لم تُضبط الإعدادات أو فشل الاتصال."""
    global _sheets_client
    if not GOOGLE_SHEET_ID or not GOOGLE_SERVICE_ACCOUNT_JSON:
        return None
    try:
        if _sheets_client is None:
            creds_info = json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)
            creds = Credentials.from_service_account_info(
                creds_info,
                scopes=["https://www.googleapis.com/auth/spreadsheets"],
            )
            _sheets_client = gspread.authorize(creds)
        return _sheets_client.open_by_key(GOOGLE_SHEET_ID).sheet1
    except Exception as e:
        log.error("Google Sheets connection error: %s", e)
        return None

def log_to_sheet(whatsapp_phone: str, name: str, national_id: str, contact_phone: str) -> None:
    sheet = get_sheet()
    if sheet is None:
        log.warning("Google Sheets غير مُفعّل - تم تخطي حفظ الصف")
        return
    try:
        sheet.append_row([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            whatsapp_phone,
            name,
            national_id,
            contact_phone,
        ])
        log.info("تم حفظ صف جديد في Google Sheet لـ %s", whatsapp_phone)
    except Exception as e:
        log.error("فشل حفظ الصف في Google Sheet: %s", e)

# ─── Gmail Notification (عبر SMTP بكلمة مرور تطبيق) ──────────────────────────
def send_notification_email(whatsapp_phone: str, name: str, national_id: str, contact_phone: str) -> None:
    if not (SMTP_EMAIL and SMTP_APP_PASSWORD and NOTIFY_EMAIL):
        log.warning("إعدادات البريد غير مُفعّلة")
        return

    body = (...) # (نفس النص الخاص بك)
    msg = MIMEText(body, _charset="utf-8")
    msg["Subject"] = f"عميل جديد يحتاج للتواصل - {name}"
    msg["From"] = SMTP_EMAIL
    msg["To"] = NOTIFY_EMAIL

    try:
        # التغيير هنا: استخدام SMTP بدلاً من SMTP_SSL والمنفذ 587
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls() # تفعيل التشفير
        server.login(SMTP_EMAIL, SMTP_APP_PASSWORD)
        server.send_message(msg)
        server.quit()
        log.info("تم إرسال إشعار البريد لـ %s", whatsapp_phone)
    except Exception as e:
        log.error("فشل إرسال إشعار البريد: %s", e)

# ─── استخراج بيانات العميل من رسالة الختام ────────────────────────────────────
def extract_customer_data(final_message: str, last_user_input: str) -> dict | None:
    """
    الاسم ورقم الهوية يُستخرجان من رسالة الختام الثابتة الصيغة (محدّدة حرفياً في الـ System Prompt).
    رقم الجوال هو آخر رسالة أرسلها المستخدم في هذه الجولة (لأنه آخر حقل يُطلب في التسلسل).
    """
    match = re.search(r"شكراً لك يا (.+?)\.\s*صاحب الهوية \((.+?)\)", final_message)
    if not match:
        log.warning("تعذّر استخراج بيانات العميل من رسالة الختام: %s", final_message)
        return None
    return {
        "name": match.group(1).strip(),
        "national_id": match.group(2).strip(),
        "contact_phone": last_user_input.strip(),
    }

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
    # 1. إعداد الـ history
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

    # 3. إرسال الرد على واتساب
    wa_send_text(state["phone"], clean)

    return {"messages": [AIMessage(content=reply)], "done": done}

# بناء الجراف — عقدة واحدة تُنفَّذ مرة واحدة بالضبط لكل رسالة واردة
# (لا يوجد self-loop على "ai"؛ التكرار عبر خطوات المحادثة يحدث عبر طلبات Webhook المتتالية لا داخل الـ Graph)
builder = StateGraph(BotState)
builder.add_node("ai", ai_node)
builder.set_entry_point("ai")
builder.add_edge("ai", END)
graph = builder.compile()

# ─── Core Logic ───────────────────────────────────────────────────────────────
def handle_message(phone: str, user_input: str) -> None:
    lock = get_phone_lock(phone)
    with lock:
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
            final_message = state["messages"][-1].content
            data = extract_customer_data(final_message, user_input)
            if data:
                log_to_sheet(phone, data["name"], data["national_id"], data["contact_phone"])
                send_notification_email(phone, data["name"], data["national_id"], data["contact_phone"])
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
