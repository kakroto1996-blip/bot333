"""
WhatsApp AI Chatbot - FastAPI + LangGraph + OpenRouter
سريع، موثوق، ويدعم العربية بشكل كامل
"""

import os
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
WHATSAPP_FLOW_ID = os.environ.get("WHATSAPP_FLOW_ID", "")
ADMIN_PASSWORD   = os.environ.get("ADMIN_PASSWORD", "")

# ─── Google Sheets + Gmail (اختياري: البوت يعمل حتى لو لم تُضبط هذه القيم) ───
GOOGLE_SHEET_ID              = os.environ.get("GOOGLE_SHEET_ID", "")
GOOGLE_SERVICE_ACCOUNT_JSON  = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "")
RESEND_API_KEY               = os.environ.get("RESEND_API_KEY", "")
NOTIFY_FROM_EMAIL            = os.environ.get("NOTIFY_FROM_EMAIL", "onboarding@resend.dev")
NOTIFY_EMAIL                 = os.environ.get("NOTIFY_EMAIL", "")

WA_URL = f"https://graph.facebook.com/v21.0/{PHONE_NUMBER_ID}/messages"

def build_system_prompt() -> str:
    return f"""الدور: أنت مساعد ذكي ومحترف لخدمة العملاء.
سياسة اللغة: تواصل باللغة العربية الفصحى فقط.

معلومات عن الشركة (استخدمها للرد على أي استفسار عن الشركة، خدماتها، ساعات عملها، موقعها، أو سياساتها):
---
{COMPANY_INFO}
---
- إذا سأل العميل أي سؤال يتعلق بهذه المعلومات، أجب عليه بدقة بالاستناد إليها فقط، ولا تخترع معلومة غير موجودة فيها.
- إذا سأل عن شيء غير موجود في المعلومات أعلاه، أخبره بلطف أنك ستحوّله لموظف للإجابة على هذا الاستفسار تحديداً.
- إذا طلب العميل تتبع طلب أو دعماً فنياً عبر رسالة نصية حرة، أخبره بلطف أنك سترسل له نموذجاً قصيراً لتعبئة بياناته وأنه يمكنه أيضاً استخدام الأزرار في رسالة الترحيب.
- ممنوع منعاً باتاً إرسال أكثر من رسالة واحدة في كل رد من طرفك."""

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

# ─── سجل المحادثات الدائم (للوحة الإدارة) ─────────────────────────────────────
def log_message(phone: str, sender: str, content: str) -> None:
    """sender: 'customer' | 'bot' | 'admin'. يُحدّث أيضاً وقت آخر نشاط للمحادثة."""
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

# ─── Gmail Notification (عبر Resend HTTPS API - يعمل على أي خطة Railway) ────
def send_notification_email(whatsapp_phone: str, name: str, national_id: str, contact_phone: str) -> None:
    if not (RESEND_API_KEY and NOTIFY_EMAIL):
        log.warning("إعدادات البريد غير مُفعّلة - تم تخطي إرسال الإشعار")
        return

    body_html = (
        "<p>عميل جديد ينتظر التواصل:</p>"
        f"<p><b>الاسم:</b> {name}<br>"
        f"<b>رقم الهوية:</b> {national_id}<br>"
        f"<b>رقم الجوال:</b> {contact_phone}<br>"
        f"<b>رقم واتساب:</b> {whatsapp_phone}</p>"
    )

    try:
        r = requests.post(
            "https://api.resend.com/emails",
            headers={
                "Authorization": f"Bearer {RESEND_API_KEY}",
                "Content-Type": "application/json",
            },
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
        else:
            log.error("فشل إرسال إشعار البريد: %s", r.text)
    except Exception as e:
        log.error("فشل إرسال إشعار البريد: %s", e)

# ─── معالجة بيانات الفورم (WhatsApp Flow) ─────────────────────────────────────
def handle_flow_submission(phone: str, form_data: dict) -> None:
    """يُستدعى عند اكتمال العميل لنموذج الفورم - البيانات تصل منظمة جاهزة، بدون أي تفسير نصي."""
    name          = str(form_data.get("full_name", "")).strip()
    national_id   = str(form_data.get("national_id", "")).strip()
    contact_phone = str(form_data.get("phone_number", "")).strip()

    thank_you = (
        f"شكراً لك يا {name}. صاحب الهوية ({national_id}) لقد تم استلام بياناتك بنجاح. "
        "نحن نقدر تعاونك معنا. سوف يتم تحويلك للموظف بأسرع وقت."
    )
    wa_send_text(phone, thank_you)
    log_message(phone, "bot", thank_you)

    log_to_sheet(phone, name, national_id, contact_phone)
    send_notification_email(phone, name, national_id, contact_phone)

    # تحويل لموظف: البوت يتوقف عن الرد التلقائي على هذا العميل من الآن، والرد يصبح يدوياً من لوحة الإدارة
    set_conversation_status(phone, "handed_off", name=name, national_id=national_id, contact_phone=contact_phone)
    delete_session(phone)  # تنظيف أي محادثة FAQ سابقة مرتبطة بهذا العميل

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

def wa_send_flow(to: str) -> None:
    if not WHATSAPP_FLOW_ID:
        log.error("WHATSAPP_FLOW_ID غير مضبوط - تعذّر إرسال الفورم")
        wa_send_text(to, "عذراً، حدث خلل مؤقت. سيتم تحويلك لموظف للمساعدة.")
        return

    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "interactive",
        "interactive": {
            "type": "flow",
            "header": {"type": "text", "text": "بيانات التواصل"},
            "body": {"text": "من فضلك أكمل النموذج التالي لمتابعة طلبك."},
            "footer": {"text": "يستغرق أقل من دقيقة"},
            "action": {
                "name": "flow",
                "parameters": {
                    "flow_message_version": "3",
                    "flow_id": WHATSAPP_FLOW_ID,
                    "flow_cta": "تعبئة البيانات",
                    "flow_action": "navigate",
                    "flow_action_payload": {"screen": "CUSTOMER_INFO"},
                },
            },
        },
    }
    r = requests.post(WA_URL, json=payload, headers=_wa_headers(), timeout=10)
    if not r.ok:
        log.error("WhatsApp flow send error: %s", r.text)
    else:
        log_message(to, "bot", "[فورم] تم إرسال نموذج بيانات التواصل")

# ─── LangGraph State ──────────────────────────────────────────────────────────
class BotState(TypedDict):
    messages: Annotated[list, add_messages]
    phone: str
    done: bool

def ai_node(state: BotState) -> BotState:
    # 1. إعداد الـ history
    history = [{"role": "system", "content": build_system_prompt()}]
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
    log_message(state["phone"], "bot", clean)

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
            log_message(from_number, "customer", user_input)

            if get_conversation_status(from_number) == "handed_off":
                # تم تحويل هذا العميل لموظف - البوت لا يرد تلقائياً، فقط يسجّل الرسالة لتظهر في اللوحة
                pass
            else:
                background.add_task(handle_message, from_number, user_input)

        elif msg_type == "interactive":
            interactive_type = message["interactive"].get("type")

            if interactive_type == "button_reply":
                button_id    = message["interactive"]["button_reply"]["id"]
                button_title = message["interactive"]["button_reply"]["title"]
                log_message(from_number, "customer", f"[زر] {button_title}")

                if button_id in ("order_tracking", "technical_support"):
                    # عند ضغط أي من الزرين، أرسل الفورم مباشرة بدل سؤال الموديل
                    background.add_task(wa_send_flow, from_number)
                else:
                    background.add_task(handle_message, from_number, button_title)

            elif interactive_type == "nfm_reply":
                # العميل أكمل الفورم وضغط إرسال - البيانات تصل منظمة جاهزة
                form_data = json.loads(message["interactive"]["nfm_reply"]["response_json"])
                log_message(from_number, "customer", f"[فورم مكتمل] {form_data}")
                background.add_task(handle_flow_submission, from_number, form_data)

            else:
                return JSONResponse({"status": "unsupported_interactive"})

        else:
            return JSONResponse({"status": "unsupported"})

        # ← الرد على WhatsApp فوراً بـ 200، ثم المعالجة في الخلفية (أُضيفت أعلاه لكل حالة)

    except (KeyError, IndexError) as e:
        log.warning("Parse error: %s", e)

    return JSONResponse({"status": "ok"})


@app.get("/health")
def health():
    return {"status": "running", "model": MODEL}


# ─── لوحة الإدارة (Admin Dashboard) ───────────────────────────────────────────
def _admin_token() -> str:
    return hashlib.sha256(ADMIN_PASSWORD.encode()).hexdigest()

def _is_admin(request: Request) -> bool:
    return bool(ADMIN_PASSWORD) and request.cookies.get("admin_token") == _admin_token()

LOGIN_HTML = """<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>تسجيل الدخول</title>
<style>
  body{margin:0;font-family:system-ui,-apple-system,Segoe UI,sans-serif;background:#0f1115;color:#e6e6e6;display:flex;align-items:center;justify-content:center;height:100vh}
  .card{background:#171a21;padding:32px;border-radius:14px;width:320px;box-shadow:0 4px 24px rgba(0,0,0,.4)}
  h1{font-size:18px;margin:0 0 20px;font-weight:700}
  input{width:100%;padding:11px;border-radius:8px;border:1px solid #2a2e38;background:#0f1115;color:#e6e6e6;margin-bottom:14px;box-sizing:border-box;font-size:14px}
  button{width:100%;padding:11px;border-radius:8px;border:none;background:#3b82f6;color:#fff;font-weight:600;cursor:pointer;font-size:14px}
  button:hover{background:#2563eb}
  .err{color:#f87171;font-size:13px;margin-bottom:12px}
</style>
</head>
<body>
  <form class="card" method="post" action="/admin/login">
    <h1>🔒 لوحة إدارة المحادثات</h1>
    __ERROR__
    <input type="password" name="password" placeholder="كلمة المرور" autofocus required>
    <button type="submit">دخول</button>
  </form>
</body>
</html>"""

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>لوحة المحادثات</title>
<style>
  *{box-sizing:border-box}
  body{margin:0;font-family:system-ui,-apple-system,Segoe UI,sans-serif;background:#0f1115;color:#e6e6e6;height:100vh;overflow:hidden}
  .app{display:flex;height:100vh}
  .list-pane{width:340px;border-left:1px solid #20242c;display:flex;flex-direction:column;background:#13151b}
  .list-header{padding:14px 16px;border-bottom:1px solid #20242c;font-weight:700;font-size:15px}
  .tabs{display:flex;gap:6px;padding:10px 12px;border-bottom:1px solid #20242c}
  .tab{flex:1;text-align:center;padding:6px 4px;border-radius:6px;font-size:12px;cursor:pointer;background:#1c1f27;color:#9aa0ab}
  .tab.active{background:#3b82f6;color:#fff}
  .conv-list{flex:1;overflow-y:auto}
  .conv-item{padding:12px 16px;border-bottom:1px solid #1b1e25;cursor:pointer;display:flex;flex-direction:column;gap:4px}
  .conv-item:hover{background:#191c23}
  .conv-item.selected{background:#1d2330}
  .conv-top{display:flex;justify-content:space-between;align-items:center}
  .conv-name{font-weight:600;font-size:14px}
  .conv-time{font-size:11px;color:#6b7280}
  .badge{font-size:10px;padding:2px 8px;border-radius:20px;font-weight:600;white-space:nowrap}
  .badge-bot{background:#1e3a2f;color:#4ade80}
  .badge-handed_off{background:#3a2e1e;color:#fbbf24}
  .badge-closed{background:#2a2d35;color:#9aa0ab}
  .conv-phone{font-size:12px;color:#8b91a0}
  .thread-pane{flex:1;display:flex;flex-direction:column}
  .thread-header{padding:14px 18px;border-bottom:1px solid #20242c;display:flex;justify-content:space-between;align-items:center}
  .thread-title{font-weight:700;font-size:15px}
  .thread-actions{display:flex;gap:8px}
  .thread-actions button{font-size:12px;padding:6px 12px;border-radius:7px;border:1px solid #2a2e38;background:#1c1f27;color:#e6e6e6;cursor:pointer}
  .thread-actions button:hover{background:#252933}
  .messages{flex:1;overflow-y:auto;padding:18px;display:flex;flex-direction:column;gap:10px}
  .msg{max-width:65%;padding:9px 13px;border-radius:12px;font-size:14px;line-height:1.5;white-space:pre-wrap}
  .msg.customer{align-self:flex-start;background:#1c1f27;border-bottom-left-radius:3px}
  .msg.bot{align-self:flex-end;background:#1e3a5f;border-bottom-right-radius:3px}
  .msg.admin{align-self:flex-end;background:#2f6b3f;border-bottom-right-radius:3px}
  .msg-meta{font-size:10px;color:#6b7280;margin-top:3px}
  .composer{display:flex;gap:8px;padding:14px;border-top:1px solid #20242c}
  .composer textarea{flex:1;resize:none;border-radius:10px;border:1px solid #2a2e38;background:#171a21;color:#e6e6e6;padding:10px 12px;font-size:14px;font-family:inherit;height:42px}
  .composer button{padding:0 18px;border-radius:10px;border:none;background:#3b82f6;color:#fff;font-weight:600;cursor:pointer}
  .composer button:hover{background:#2563eb}
  .empty{flex:1;display:flex;align-items:center;justify-content:center;color:#6b7280;font-size:14px}
  .disabled-note{padding:8px 18px;background:#3a2e1e;color:#fbbf24;font-size:12px;text-align:center}
</style>
</head>
<body>
<div class="app">
  <div class="list-pane">
    <div class="list-header">💬 المحادثات</div>
    <div class="tabs">
      <div class="tab active" data-f="all">الكل</div>
      <div class="tab" data-f="handed_off">يحتاج رد</div>
      <div class="tab" data-f="bot">نشط (بوت)</div>
      <div class="tab" data-f="closed">مغلق</div>
    </div>
    <div class="conv-list" id="convList"></div>
  </div>
  <div class="thread-pane">
    <div id="threadEmpty" class="empty">اختر محادثة من القائمة</div>
    <div id="threadView" style="display:none;flex:1;display:flex;flex-direction:column">
      <div class="thread-header">
        <div>
          <div class="thread-title" id="threadTitle">—</div>
          <div class="conv-phone" id="threadPhone">—</div>
        </div>
        <div class="thread-actions">
          <button onclick="setStatus('bot')">🤖 أعد للبوت</button>
          <button onclick="setStatus('closed')">✅ إغلاق</button>
        </div>
      </div>
      <div class="messages" id="messages"></div>
      <div class="composer">
        <textarea id="replyBox" placeholder="اكتب رداً للعميل..." onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();sendReply();}"></textarea>
        <button onclick="sendReply()">إرسال</button>
      </div>
    </div>
  </div>
</div>
<script>
let currentPhone = null;
let currentFilter = 'all';
let allConvs = [];

function badgeLabel(s){return {bot:'🤖 يرد البوت', handed_off:'🧑‍💼 يحتاج رد', closed:'✅ مغلق'}[s] || s;}

async function loadConversations(){
  const r = await fetch('/admin/api/conversations');
  if(r.status === 401){ location.href = '/admin/login'; return; }
  const data = await r.json();
  allConvs = data.conversations;
  renderList();
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
      <div class="conv-phone">${c.phone}</div>
      <div class="conv-time">${c.updated_at || ''}</div>
    </div>
  `).join('') || '<div style="padding:20px;color:#6b7280;font-size:13px">لا توجد محادثات</div>';
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
  document.getElementById('threadPhone').textContent = phone;
  renderList();
  await loadMessages();
}

async function loadMessages(){
  if(!currentPhone) return;
  const r = await fetch('/admin/api/messages/' + encodeURIComponent(currentPhone));
  if(r.status === 401){ location.href = '/admin/login'; return; }
  const data = await r.json();
  const box = document.getElementById('messages');
  const wasAtBottom = box.scrollTop + box.clientHeight >= box.scrollHeight - 30;
  box.innerHTML = data.messages.map(m => `
    <div class="msg ${m.sender}">${escapeHtml(m.content)}<div class="msg-meta">${m.created_at}</div></div>
  `).join('');
  if(wasAtBottom) box.scrollTop = box.scrollHeight;
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
  await loadMessages();
  await loadConversations();
}

async function setStatus(status){
  if(!currentPhone) return;
  await fetch('/admin/api/status/' + encodeURIComponent(currentPhone), {
    method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({status})
  });
  await loadConversations();
}

loadConversations();
setInterval(loadConversations, 5000);
setInterval(loadMessages, 3000);
</script>
</body>
</html>"""

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
    return HTMLResponse(
        LOGIN_HTML.replace("__ERROR__", '<div class="err">كلمة المرور غير صحيحة</div>'),
        status_code=401,
    )

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
