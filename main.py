"""
WhatsApp AI Chatbot - FastAPI + LangGraph + OpenRouter
دمج نظام الأسئلة المتسلسلة (main 2) مع لوحة الإدارة والداشبورد (main 4)
"""

import os
import re
import json
import hashlib
import logging
import sqlite3
import threading
import requests
from datetime import datetime
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Annotated, TypedDict
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import PlainTextResponse, JSONResponse, HTMLResponse, RedirectResponse
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
COMPANY_INFO     = os.environ.get("COMPANY_INFO", "لا تتوفر معلومات عن الشركة حالياً.")
ADMIN_PASSWORD   = os.environ.get("ADMIN_PASSWORD", "")

# ─── Google Sheets + Gmail (اختياري) ──────────────────────────────────────────
GOOGLE_SHEET_ID              = os.environ.get("GOOGLE_SHEET_ID", "")
GOOGLE_SERVICE_ACCOUNT_JSON  = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "")
RESEND_API_KEY               = os.environ.get("RESEND_API_KEY", "")
NOTIFY_FROM_EMAIL            = os.environ.get("NOTIFY_FROM_EMAIL", "onboarding@resend.dev")
NOTIFY_EMAIL                 = os.environ.get("NOTIFY_EMAIL", "")

WA_URL = f"https://graph.facebook.com/v21.0/{PHONE_NUMBER_ID}/messages"

# ─── System Prompt (نظام الأسئلة المتسلسلة من main 2 مع معلومات الشركة) ──────
def build_system_prompt() -> str:
    return f"""الدور: أنت مساعد ذكي ومحترف لخدمة العملاء.
سياسة اللغة: تواصل باللغة العربية الفصحى فقط.

معلومات عن الشركة (استخدمها للرد على استفسارات العميل العامة قبل أو أثناء جمع البيانات):
---
{COMPANY_INFO}
---

مهمتك الأساسية: عند طلب تتبع طلب أو دعم فني، يجب جمع 3 معلومات خطوة بخطوة (معلومة واحدة في كل رسالة):
1. الاسم الثلاثي
2. رقم الهوية
3. رقم الجوال

قواعد صارمة (يجب اتباعها بدقة):
- لا تطلب أكثر من معلومة واحدة في كل رد.
- إذا قام المستخدم بتزويدك بمعلومة، لا تطلبها مجدداً، وانتقل فوراً للخطوة التالية.
- لا تكرر السؤال السابق إذا أجاب المستخدم عليه بالفعل.
- إذا أرسل المستخدم زر (تتبع الطلب أو الدعم الفني)، ابدأ بطلب الاسم الثلاثي فوراً.
- التزم بالهدوء والاحترافية: إذا أرسل المستخدم رسالة لا تحتوي على المعلومة المطلوبة، اطلبها منه بأسلوب مهذب مرة واحدة فقط.
- ممنوع منعاً باتاً إرسال أكثر من رسالة واحدة في كل رد من طرفك.

بعد جمع المعلومات الثلاث أرسل هذه الرسالة حرفياً ودون أي تعديل:
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
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            phone TEXT PRIMARY KEY,
            history TEXT NOT NULL DEFAULT '[]',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            phone TEXT PRIMARY KEY,
            status TEXT NOT NULL DEFAULT 'bot',
            customer_name TEXT,
            national_id TEXT,
            contact_phone TEXT,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS messages_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            phone TEXT NOT NULL,
            sender TEXT NOT NULL,
            content TEXT NOT NULL,
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

# ─── سجل المحادثات الدائم والداشبورد ──────────────────────────────────────────
def log_message(phone: str, sender: str, content: str) -> None:
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT INTO messages_log (phone, sender, content) VALUES (?,?,?)",
        (phone, sender, content),
    )
    con.execute(
        "INSERT INTO conversations (phone, updated_at) VALUES (?, CURRENT_TIMESTAMP) "
        "ON CONFLICT(phone) DO UPDATE SET updated_at=CURRENT_TIMESTAMP",
        (phone,),
    )
    con.commit()
    con.close()

def get_conversation_status(phone: str) -> str:
    con = sqlite3.connect(DB_PATH)
    row = con.execute("SELECT status FROM conversations WHERE phone=?", (phone,)).fetchone()
    con.close()
    return row[0] if row else "bot"

def set_conversation_status(
    phone: str, status: str,
    name: str | None = None, national_id: str | None = None, contact_phone: str | None = None,
) -> None:
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT INTO conversations (phone, status, customer_name, national_id, contact_phone, updated_at) "
        "VALUES (?,?,?,?,?, CURRENT_TIMESTAMP) "
        "ON CONFLICT(phone) DO UPDATE SET "
        "status=excluded.status, "
        "customer_name=COALESCE(excluded.customer_name, conversations.customer_name), "
        "national_id=COALESCE(excluded.national_id, conversations.national_id), "
        "contact_phone=COALESCE(excluded.contact_phone, conversations.contact_phone), "
        "updated_at=CURRENT_TIMESTAMP",
        (phone, status, name, national_id, contact_phone),
    )
    con.commit()
    con.close()

def list_conversations() -> list[dict]:
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT phone, status, customer_name, national_id, contact_phone, updated_at "
        "FROM conversations ORDER BY updated_at DESC"
    ).fetchall()
    con.close()
    return [
        {
            "phone": r[0], "status": r[1], "customer_name": r[2],
            "national_id": r[3], "contact_phone": r[4], "updated_at": r[5],
        }
        for r in rows
    ]

def get_messages(phone: str) -> list[dict]:
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT sender, content, created_at FROM messages_log WHERE phone=? ORDER BY id ASC",
        (phone,),
    ).fetchall()
    con.close()
    return [{"sender": r[0], "content": r[1], "created_at": r[2]} for r in rows]

# ─── Per-Phone Locks ──────────────────────────────────────────────────────────
_phone_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
_locks_guard = threading.Lock()

def get_phone_lock(phone: str) -> threading.Lock:
    with _locks_guard:
        return _phone_locks[phone]

# ─── Google Sheets ────────────────────────────────────────────────────────────
_sheets_client = None

def get_sheet():
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

# ─── Email Notification ───────────────────────────────────────────────────────
def send_notification_email(whatsapp_phone: str, name: str, national_id: str, contact_phone: str) -> None:
    if not (RESEND_API_KEY and NOTIFY_EMAIL):
        return

    body_html = (
        "<p>عميل جديد ينتظر التواصل (تم جمع بياناته عبر البوت التلقائي):</p>"
        f"<p><b>الاسم:</b> {name}<br>"
        f"<b>رقم الهوية:</b> {national_id}<br>"
        f"<b>رقم الجوال:</b> {contact_phone}<br>"
        f"<b>رقم واتساب:</b> {whatsapp_phone}</p>"
    )

    try:
        r = requests.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {RESEND_API_KEY}", "Content-Type": "application/json"},
            json={
                "from": NOTIFY_FROM_EMAIL,
                "to": [NOTIFY_EMAIL],
                "subject": f"عميل جديد يحتاج للتواصل - {name}",
                "html": body_html,
            },
            timeout=10,
        )
        if r.ok:
            log.info("تم إرسال إشعار البريد لـ %s", whatsapp_phone)
    except Exception as e:
        log.error("فشل إرسال إشعار البريد: %s", e)

# ─── استخراج بيانات العميل تلقائياً عند انتهاء المحادثة النصية ─────────────────
def extract_customer_data(final_message: str, last_user_input: str) -> dict | None:
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
    else:
        log_message(to, "bot", "[أزرار] مرحباً بك! 👋 كيف يمكنني مساعدتك اليوم؟")

# ─── LangGraph State ──────────────────────────────────────────────────────────
class BotState(TypedDict):
    messages: Annotated[list, add_messages]
    phone: str
    done: bool

def ai_node(state: BotState) -> BotState:
    history = [{"role": "system", "content": build_system_prompt()}]
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            history.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            history.append({"role": "assistant", "content": msg.content})

    response = ai.chat.completions.create(model=MODEL, messages=history)
    reply = response.choices[0].message.content or ""

    done = "[DONE]" in reply
    clean = reply.replace("[DONE]", "").strip()

    wa_send_text(state["phone"], clean)
    log_message(state["phone"], "bot", clean)

    return {"messages": [AIMessage(content=reply)], "done": done}

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
            wa_send_buttons(phone)
            welcome_msg = "مرحباً بك! 👋 كيف يمكنني مساعدتك اليوم؟"
            new_history = [{"role": "assistant", "content": welcome_msg}]
            save_session(phone, new_history)
            return

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
                # تخزين البيانات في الداشبورد وتغيير الحالة لتنبيه الموظف
                set_conversation_status(
                    phone, "handed_off", 
                    name=data["name"], 
                    national_id=data["national_id"], 
                    contact_phone=data["contact_phone"]
                )
                log_to_sheet(phone, data["name"], data["national_id"], data["contact_phone"])
                send_notification_email(phone, data["name"], data["national_id"], data["contact_phone"])
            else:
                set_conversation_status(phone, "handed_off")
                
            delete_session(phone)  # تصفير الجلسة التلقائية بعد تسليمها للموظف
        else:
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
    params = request.query_params
    if params.get("hub.mode") == "subscribe" and params.get("hub.verify_token") == VERIFY_TOKEN:
        log.info("Webhook verified ✓")
        return PlainTextResponse(params.get("hub.challenge", ""))
    return PlainTextResponse("Forbidden", status_code=403)


@app.post("/webhook")
async def webhook(request: Request, background: BackgroundTasks):
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

        if from_number == display:
            return JSONResponse({"status": "self"})

        msg_type = message.get("type")
        
        # إذا كانت المحادثة محولة للموظف (handed_off)، لن يقوم البوت بالرد التلقائي، فقط يسجل الرسائل في الداشبورد
        is_handed_off = get_conversation_status(from_number) == "handed_off"

        if msg_type == "text":
            user_input = message["text"]["body"].strip()
            log_message(from_number, "customer", user_input)

            if not is_handed_off:
                background.add_task(handle_message, from_number, user_input)

        elif msg_type == "interactive":
            interactive_type = message["interactive"].get("type")

            if interactive_type == "button_reply":
                button_title = message["interactive"]["button_reply"]["title"]
                log_message(from_number, "customer", f"[زر] {button_title}")

                if not is_handed_off:
                    # نرسل عنوان الزر مباشرة للموديل ليبدأ بطلب الاسم الثلاثي بناءً على شروط الـ System Prompt
                    background.add_task(handle_message, from_number, button_title)

            else:
                return JSONResponse({"status": "unsupported_interactive"})
        else:
            return JSONResponse({"status": "unsupported"})

    except (KeyError, IndexError) as e:
        log.warning("Parse error: %s", e)

    return JSONResponse({"status": "ok"})


@app.get("/health")
def health():
    return {"status": "running", "model": MODEL}


# ─── لوحة الإدارة (Admin Dashboard HTML) ──────────────────────────────────────
def _admin_token() -> str:
    return hashlib.sha256(ADMIN_PASSWORD.encode()).hexdigest()

def _is_admin(request: Request) -> bool:
    return bool(ADMIN_PASSWORD) and request.cookies.get("admin_token") == _admin_token()

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>مركز إدارة المحادثات الذكي</title>
<!-- استيراد الخطوط والأيقونات لجعل التصميم احترافي -->
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<style>
  :root {
    --bg-main: #0b0f17;
    --bg-sidebar: #111827;
    --bg-card: #1f2937;
    --bg-chat: #0f172a;
    --primary: #3b82f6;
    --primary-hover: #2563eb;
    --text-main: #f3f4f6;
    --text-muted: #9ca3af;
    --border: #374151;
    --badge-bot: rgba(16, 185, 129, 0.15);
    --badge-bot-text: #10b981;
    --badge-user: rgba(245, 158, 11, 0.15);
    --badge-user-text: #f59e0b;
    --badge-close: rgba(107, 114, 128, 0.2);
    --badge-close-text: #9ca3af;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; font-family: system-ui, -apple-system, sans-serif; }
  body { background: var(--bg-main); color: var(--text-main); height: 100vh; overflow: hidden; display: flex; flex-direction: column; }
  
  /* الهيدر العلوي للنظام */
  .navbar { background: var(--bg-sidebar); height: 60px; display: flex; align-items: center; justify-content: space-between; padding: 0 24px; border-bottom: 1px solid var(--border); z-index: 100; }
  .navbar-brand { display: flex; align-items: center; gap: 12px; font-size: 18px; font-weight: 700; color: #fff; }
  .navbar-brand i { color: var(--primary); font-size: 22px; }
  .logout-btn { color: #f87171; text-decoration: none; font-size: 14px; display: flex; align-items: center; gap: 6px; padding: 6px 12px; border-radius: 6px; transition: 0.2s; }
  .logout-btn:hover { background: rgba(248, 113, 113, 0.1); }

  /* الهيكل الأساسي */
  .app-container { display: flex; flex: 1; height: calc(100vh - 60px); overflow: hidden; position: relative; }
  
  /* القائمة الجانبية (المحادثات) */
  .sidebar { width: 360px; min-width: 360px; background: var(--bg-sidebar); border-left: 1px solid var(--border); display: flex; flex-direction: column; height: 100%; }
  .sidebar-header { padding: 16px; border-bottom: 1px solid var(--border); }
  .tabs { display: flex; gap: 6px; background: rgba(0,0,0,0.2); padding: 4px; border-radius: 8px; }
  .tab { flex: 1; text-align: center; padding: 8px 4px; border-radius: 6px; font-size: 13px; font-weight: 600; cursor: pointer; color: var(--text-muted); transition: 0.2s; user-select: none; }
  .tab.active { background: var(--bg-card); color: #fff; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
  
  .conv-list { flex: 1; overflow-y: auto; padding: 8px; }
  .conv-item { padding: 14px; border-radius: 10px; cursor: pointer; display: flex; flex-direction: column; gap: 6px; margin-bottom: 6px; transition: 0.2s; border: 1px solid transparent; }
  .conv-item:hover { background: rgba(255,255,255,0.02); }
  .conv-item.selected { background: var(--bg-card); border-color: var(--border); }
  .conv-top { display: flex; justify-content: space-between; align-items: center; }
  .conv-name { font-weight: 600; font-size: 14px; color: #fff; max-width: 180px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .conv-time { font-size: 11px; color: var(--text-muted); }
  .conv-phone { font-size: 12px; color: var(--text-muted); display: flex; align-items: center; gap: 6px; }
  
  /* الشارات (Badges) */
  .badge { font-size: 11px; padding: 4px 10px; border-radius: 20px; font-weight: 600; display: inline-flex; align-items: center; gap: 4px; }
  .badge-bot { background: var(--badge-bot); color: var(--badge-bot-text); }
  .badge-handed_off { background: var(--badge-user); color: var(--badge-user-text); }
  .badge-closed { background: var(--badge-close); color: var(--badge-close-text); }

  /* منطقة المحادثة */
  .chat-view { flex: 1; display: flex; flex-direction: column; background: var(--bg-chat); height: 100%; position: relative; }
  
  /* هيدر المحادثة النشطة */
  .chat-header { padding: 16px 24px; border-bottom: 1px solid var(--border); background: var(--bg-sidebar); display: flex; justify-content: space-between; align-items: center; }
  .chat-user-info h3 { font-size: 16px; font-weight: 700; color: #fff; margin-bottom: 2px; }
  .chat-user-info span { font-size: 13px; color: var(--text-muted); }
  .chat-actions { display: flex; gap: 8px; }
  .chat-actions button { font-size: 13px; padding: 8px 14px; border-radius: 8px; border: 1px solid var(--border); background: var(--bg-card); color: #fff; cursor: pointer; font-weight: 600; display: flex; align-items: center; gap: 6px; transition: 0.2s; }
  .chat-actions button:hover { background: var(--border); }

  /* صندوق البيانات المستخرجة المستوحى من Chatwoot */
  .meta-sidebar { background: rgba(59, 130, 246, 0.05); border: 1px dashed rgba(59, 130, 246, 0.3); padding: 12px 20px; margin: 16px 24px 0 24px; border-radius: 10px; display: flex; gap: 24px; align-items: center; font-size: 13px; }
  .meta-item { display: flex; align-items: center; gap: 8px; color: var(--text-main); }
  .meta-item i { color: var(--primary); font-size: 15px; }

  /* منطقة الرسائل */
  .messages-container { flex: 1; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 12px; }
  .msg-wrapper { display: flex; flex-direction: column; width: 100%; }
  .msg { max-width: 60%; padding: 12px 16px; font-size: 14.5px; line-height: 1.5; white-space: pre-wrap; word-break: break-word; position: relative; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
  
  .msg.customer { align-self: flex-start; background: var(--bg-sidebar); color: #fff; border-radius: 14px 14px 0px 14px; border: 1px solid var(--border); }
  .msg.bot { align-self: flex-end; background: #1e293b; color: #e2e8f0; border-radius: 14px 14px 14px 0px; border: 1px solid rgba(255,255,255,0.05); }
  .msg.admin { align-self: flex-end; background: #1e3a8a; color: #fff; border-radius: 14px 14px 14px 0px; }
  
  .msg-meta { font-size: 10px; color: var(--text-muted); margin-top: 6px; text-align: left; display: flex; align-items: center; justify-content: flex-end; gap: 4px; }
  .msg-meta i { font-size: 12px; color: #3b82f6; }

  /* صندوق إرسال الرد المتطور */
  .chat-composer { padding: 16px 24px; border-top: 1px solid var(--border); background: var(--bg-sidebar); display: flex; gap: 12px; align-items: center; position: sticky; bottom: 0; }
  .composer-wrapper { flex: 1; position: relative; display: flex; align-items: center; }
  .chat-composer textarea { width: 100%; resize: none; border-radius: 24px; border: 1px solid var(--border); background: var(--bg-main); color: #fff; padding: 12px 20px; font-size: 14px; outline: none; transition: 0.2s; height: 46px; line-height: 20px; overflow-y: hidden; }
  .chat-composer textarea:focus { border-color: var(--primary); box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2); }
  
  .send-btn { background: var(--primary); color: #fff; border: none; width: 44px; height: 44px; border-radius: 50%; cursor: pointer; display: flex; align-items: center; justify-content: center; font-size: 16px; transition: 0.2s; min-width: 44px; }
  .send-btn:hover { background: var(--primary-hover); transform: scale(1.05); }

  /* واجهة عدم اختيار محادثة */
  .empty-state { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; color: var(--text-muted); gap: 16px; background: var(--bg-chat); }
  .empty-state i { font-size: 48px; color: var(--border); }
</style>
</head>
<body>

<nav class="navbar">
  <div class="navbar-brand">
    <i class="fa-solid fa-comments-dollar"></i>
    <span>مركز إدارة المحادثات الذكي</span>
  </div>
  <a href="/admin/logout" class="logout-btn">
    <i class="fa-solid fa-right-from-bracket"></i> خروج
  </a>
</nav>

<div class="app-container">
  <!-- القائمة الجانبية للمحادثات -->
  <div class="sidebar">
    <div class="sidebar-header">
      <div class="tabs">
        <div class="tab active" data-f="all">الكل</div>
        <div class="tab" data-f="handed_off">يحتاج رد</div>
        <div class="tab" data-f="bot">نشط (بوت)</div>
        <div class="tab" data-f="closed">مغلق</div>
      </div>
    </div>
    <div class="conv-list" id="convList"></div>
  </div>

  <!-- منطقة شاشة المحادثة -->
  <div class="chat-view">
    <div id="threadEmpty" class="empty-state">
      <i class="fa-regular fa-message"></i>
      <p>اختر محادثة من القائمة الجانبية لبدء المتابعة</p>
    </div>
    
    <div id="threadView" style="display:none; flex-direction:column; height:100%; overflow:hidden;">
      <!-- هيدر شاشة الشات -->
      <div class="chat-header">
        <div class="chat-user-info">
          <h3 id="threadTitle">—</h3>
          <span id="threadPhone">—</span>
        </div>
        <div class="chat-actions">
          <button onclick="setStatus('bot')"><i class="fa-solid fa-robot"></i> تفعيل البوت</button>
          <button onclick="setStatus('closed')"><i class="fa-solid fa-circle-check"></i> إغلاق التذكرة</button>
        </div>
      </div>
      
      <!-- الصندوق الجديد لعرض بيانات العميل -->
      <div id="customerMeta" class="meta-sidebar" style="display:none;"></div>
      
      <!-- حاوية الرسائل المتدفقة -->
      <div class="messages-container" id="messages"></div>
      
      <!-- كcomposer صندوق الكتابة التفاعلي -->
      <div class="chat-composer">
        <div class="composer-wrapper">
          <textarea id="replyBox" placeholder="اكتب رداً مخصصاً للعميل..." onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();sendReply();}"></textarea>
        </div>
        <button class="send-btn" onclick="sendReply()"><i class="fa-solid fa-paper-plane"></i></button>
      </div>
    </div>
  </div>
</div>

<script>
let currentPhone = null;
let currentFilter = 'all';
let allConvs = [];

function badgeLabel(s){return {bot:'🤖 تلقائي', handed_off:'🧑‍💼 يحتاج موظف', closed:'✅ مغلق'}[s] || s;}

async function loadConversations(forceRender = false){
  const r = await fetch('/admin/api/conversations');
  if(r.status === 401){ location.href = '/admin/login'; return; }
  const data = await r.json();
  allConvs = data.conversations;
  renderList();
  if(currentPhone) updateMetadataBox();
}

function renderList(){
  const list = document.getElementById('convList');
  const filtered = currentFilter === 'all' ? allConvs : allConvs.filter(c => c.status === currentFilter);
  
  list.innerHTML = filtered.map(c => `
    <div class="conv-item ${c.phone===currentPhone?'selected':''}" onclick="openConv('${c.phone}')">
      <div class="conv-top">
        <span class="conv-name">${c.customer_name || c.phone}</span>
        <span class="badge badge-${c.status}">${badgeLabel(c.status)}</span>
      </div>
      <div class="conv-phone"><i class="fa-brands fa-whatsapp"></i> ${c.phone}</div>
      <div class="conv-time">${c.updated_at ? c.updated_at.substring(11,16) : ''}</div>
    </div>
  `).join('') || '<div style="padding:20px;color:var(--text-muted);font-size:13px;text-align:center;">لا توجد محادثات في هذا القسم</div>';
}

function updateMetadataBox(){
  const conv = allConvs.find(c => c.phone === currentPhone);
  const metaBox = document.getElementById('customerMeta');
  if(conv && (conv.customer_name || conv.national_id || conv.contact_phone)){
    metaBox.style.display = 'flex';
    metaBox.innerHTML = `
      <div class="meta-item"><i class="fa-solid fa-user-tag"></i> <b>الاسم:</b> ${conv.customer_name || '—'}</div>
      <div class="meta-item"><i class="fa-solid fa-id-card"></i> <b>الهوية:</b> ${conv.national_id || '—'}</div>
      <div class="meta-item"><i class="fa-solid fa-mobile-button"></i> <b>الجوال:</b> ${conv.contact_phone || '—'}</div>
    `;
  } else {
    metaBox.style.display = 'none';
  }
}

document.querySelectorAll('.tab').forEach(t => t.onclick = () => {
  document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
  t.classList.add('active');
  currentFilter = t.dataset.f;
  renderList();
});

async function openConv(phone){
  currentPhone = phone;
  document.getElementById('threadEmpty').style.display = 'none';
  document.getElementById('threadView').style.display = 'flex';
  const conv = allConvs.find(c => c.phone === phone);
  document.getElementById('threadTitle').textContent = (conv && conv.customer_name) || phone;
  document.getElementById('threadPhone').textContent = "رقم الواتساب: " + phone;
  renderList();
  updateMetadataBox();
  await loadMessages(true);
}

async function loadMessages(forceScroll = false){
  if(!currentPhone) return;
  const r = await fetch('/admin/api/messages/' + encodeURIComponent(currentPhone));
  if(r.status === 401){ location.href = '/admin/login'; return; }
  const data = await r.json();
  const box = document.getElementById('messages');
  const wasAtBottom = box.scrollTop + box.clientHeight >= box.scrollHeight - 60;
  
  box.innerHTML = data.messages.map(m => {
    let checkIcon = m.sender === 'admin' ? ' <i class="fa-solid fa-check-double"></i>' : '';
    return `
      <div class="msg-wrapper">
        <div class="msg ${m.sender}">
          ${escapeHtml(m.content)}
          <div class="msg-meta">${m.created_at.substring(11,16)}${checkIcon}</div>
        </div>
      </div>
    `;
  }).join('');
  
  if(wasAtBottom || forceScroll) box.scrollTop = box.scrollHeight;
}

function escapeHtml(s){
  const d = document.createElement('div'); d.textContent = s; return d.innerHTML;
}

async function sendReply(){
  const box = document.getElementById('replyBox');
  const text = box.value.trim();
  if(!text || !currentPhone) return;
  box.value = '';
  await fetch('/admin/api/reply/' + encodeURIComponent(currentPhone), {
    method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({text})
  });
  await loadMessages(true);
  await loadConversations(true);
}

async function setStatus(status){
  if(!currentPhone) return;
  await fetch('/admin/api/status/' + encodeURIComponent(currentPhone), {
    method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({status})
  });
  await loadConversations(true);
}

loadConversations(true);
setInterval(() => loadConversations(false), 5000);
setInterval(() => loadMessages(false), 3000);
</script>
</body>
</html>"""
# ─── Admin API Endpoints ──────────────────────────────────────────────────────
@app.get("/admin/login", response_class=HTMLResponse)
def admin_login_page():
    return LOGIN_HTML.replace("__ERROR__", "")

@app.post("/admin/login")
async def admin_login(request: Request):
    if not ADMIN_PASSWORD:
        return HTMLResponse(
            LOGIN_HTML.replace("__ERROR__", '<div class="err">ADMIN_PASSWORD غير مضبوط في متغيرات البيئة</div>'),
            status_code=500,
        )
    form = await request.form()
    if form.get("password", "") == ADMIN_PASSWORD:
        resp = RedirectResponse(url="/admin", status_code=303)
        resp.set_cookie("admin_token", _admin_token(), httponly=True, samesite="lax", max_age=60 * 60 * 24 * 30)
        return resp
    return HTMLResponse(LOGIN_HTML.replace("__ERROR__", '<div class="err">كلمة المرور غير صحيحة</div>'), status_code=401)

@app.get("/admin/logout")
def admin_logout():
    resp = RedirectResponse(url="/admin/login")
    resp.delete_cookie("admin_token")
    return resp

@app.get("/admin", response_class=HTMLResponse)
def admin_dashboard(request: Request):
    if not _is_admin(request):
        return RedirectResponse(url="/admin/login")
    return DASHBOARD_HTML

@app.get("/admin/api/conversations")
def admin_api_conversations(request: Request):
    if not _is_admin(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return {"conversations": list_conversations()}

@app.get("/admin/api/messages/{phone}")
def admin_api_messages(phone: str, request: Request):
    if not _is_admin(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return {"messages": get_messages(phone)}

@app.post("/admin/api/reply/{phone}")
async def admin_api_reply(phone: str, request: Request):
    if not _is_admin(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    body = await request.json()
    text = (body.get("text") or "").strip()
    if not text:
        return JSONResponse({"error": "empty"}, status_code=400)
    
    wa_send_text(phone, text)
    log_message(phone, "admin", text)
    set_conversation_status(phone, "handed_off")
    return {"status": "sent"}

@app.post("/admin/api/status/{phone}")
async def admin_api_set_status(phone: str, request: Request):
    if not _is_admin(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    body = await request.json()
    new_status = body.get("status")
    if new_status not in ("bot", "handed_off", "closed"):
        return JSONResponse({"error": "invalid status"}, status_code=400)
    set_conversation_status(phone, new_status)
    return {"status": "ok"}
