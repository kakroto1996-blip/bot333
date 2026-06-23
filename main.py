"""
WhatsApp AI Chatbot - FastAPI + LangGraph + OpenRouter
سريع، موثوق، ويدعم العربية بشكل كامل + لوحة تحكم مصغرة مدمجة
"""

import os
import re
import json
import logging
import sqlite3
import threading
import requests
from datetime import datetime
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Annotated, TypedDict
from fastapi import FastAPI, Request, BackgroundTasks, Form
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

# ─── Google Sheets + Gmail (اختياري: البوت يعمل حتى لو لم تُضبط هذه القيم) ───
GOOGLE_SHEET_ID              = os.environ.get("GOOGLE_SHEET_ID", "")
GOOGLE_SERVICE_ACCOUNT_JSON  = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "")
RESEND_API_KEY               = os.environ.get("RESEND_API_KEY", "")
NOTIFY_FROM_EMAIL            = os.environ.get("NOTIFY_FROM_EMAIL", "onboarding@resend.dev")
NOTIFY_EMAIL                 = os.environ.get("NOTIFY_EMAIL", "")

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
    # أضفنا حقل status للتفريق بين العميل النشط مع البوت 'bot' والعميل المحول للموظف 'agent'
    con.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            phone TEXT PRIMARY KEY,
            history TEXT NOT NULL DEFAULT '[]',
            status TEXT NOT NULL DEFAULT 'bot',
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

def save_session(phone: str, history: list[dict], status: str = "bot") -> None:
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT INTO sessions (phone, history, status) VALUES (?,?,?,?) "
        "ON CONFLICT(phone) DO UPDATE SET history=excluded.history, status=excluded.status",
        (phone, json.dumps(history, ensure_ascii=False), status),
    )
    con.commit()
    con.close()

# بدلاً من الحذف الكلي مباشرة، يتم وسمها فقط كـ 'agent' لتعرض في لوحة الإدارة
def mark_as_agent(phone: str) -> None:
    con = sqlite3.connect(DB_PATH)
    con.execute("UPDATE sessions SET status='agent' WHERE phone=?", (phone,))
    con.commit()
    con.close()
    log.info("Session updated to agent status for %s", phone)

def delete_session(phone: str) -> None:
    con = sqlite3.connect(DB_PATH)
    con.execute("DELETE FROM sessions WHERE phone=?", (phone,))
    con.commit()
    con.close()
    log.info("Session deleted for %s", phone)

# دالة مساعدة جديدة للوحة التحكم لجلب العملاء المحولين
def get_all_agent_sessions() -> list[dict]:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    rows = con.execute("SELECT phone, history FROM sessions WHERE status='agent'").fetchall()
    con.close()
    
    result = []
    for r in rows:
        history = json.loads(r["history"])
        name = r["phone"]
        for msg in reversed(history):
            if msg["role"] == "assistant" and "شكراً لك يا" in msg["content"]:
                match = re.search(r"شكراً لك يا (.+?)\.", msg["content"])
                if match:
                    name = match.group(1).strip()
                    break
        result.append({"phone": r["phone"], "name": name, "history": history})
    return result

# ─── Per-Phone Locks ──────────────────────────────────────────────────────────
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
builder = StateGraph(BotState)
builder.add_node("ai", ai_node)
builder.set_entry_point("ai")
builder.add_edge("ai", END)
graph = builder.compile()

# ─── Core Logic ───────────────────────────────────────────────────────────────
def handle_message(phone: str, user_input: str) -> None:
    lock = get_phone_lock(phone)
    with lock:
        # فحص حالة المحادثة أولاً لمنع رد البوت التلقائي إذا تحولت للموظف
        con = sqlite3.connect(DB_PATH)
        row = con.execute("SELECT status FROM sessions WHERE phone=?", (phone,)).fetchone()
        con.close()
        if row and row[0] == "agent":
            # العميل يتحدث مع الموظف الآن، نقوم فقط بحفظ رسالته في السجل ليراها الموظف في اللوحة
            history = load_session(phone)
            history.append({"role": "user", "content": user_input})
            save_session(phone, history, status="agent")
            return

        history = load_session(phone)

        if not history:
            # 1. إرسال الأزرار
            wa_send_buttons(phone)

            # 2. حفظ رسالة الترحيب في الـ history فوراً لمنع تكرارها
            welcome_msg = "مرحباً بك! 👋 كيف يمكنني مساعدتك اليوم?"
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

        # حفظ المحادثة المحدّثة
        new_history = history + [
            {"role": "user",      "content": user_input},
            {"role": "assistant", "content": state["messages"][-1].content},
        ]

        if state.get("done"):
            final_message = state["messages"][-1].content
            data = extract_customer_data(final_message, user_input)
            if data:
                log_to_sheet(phone, data["name"], data["national_id"], data["contact_phone"])
                send_notification_email(phone, data["name"], data["national_id"], data["contact_phone"])
            
            # تم استبدال delete_session بـ save_session مع حالة agent لتثبيتها في اللوحة
            save_session(phone, new_history, status="agent")
            log.info("Conversation successfully completed and routed to dashboard for %s", phone)
        else:
            save_session(phone, new_history, status="bot")

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


# ─── 🎛️ قسم لوحة التحكم المضافة بالكامل أسفل الكود دون المساس بالبنية ─────────────────

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <title>لوحة الدعم المصغرة</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&display=swap" rel="stylesheet">
    <style> body { font-family: 'Cairo', sans-serif; } </style>
    <meta http-equiv="refresh" content="10">
</head>
<body class="bg-gray-100 min-h-screen flex flex-col">
    <header class="bg-blue-600 text-white p-4 shadow-md flex justify-between items-center">
        <h1 class="text-lg font-bold">📲 واجهة الموظف البشري لمتابعة محادثات واتساب</h1>
        <span class="bg-blue-700 px-3 py-1 rounded text-xs">تحديث تلقائي كل 10 ثوانٍ</span>
    </header>

    <main class="flex-1 p-4 max-w-6xl w-full mx-auto grid grid-cols-1 md:grid-cols-3 gap-4">
        <!-- القائمة اليمنى -->
        <div class="bg-white rounded-lg shadow p-3 col-span-1 overflow-y-auto max-h-[78vh]">
            <h2 class="font-bold text-gray-700 mb-3 border-b pb-1 text-sm">📥 عملاء جاهزون للرد</h2>
            {% if not sessions %}
                <p class="text-gray-400 text-center py-6 text-xs">لا توجد محادثات بانتظارك حالياً.</p>
            {% endif %}
            {% for s in sessions %}
                <a href="?active={{ s.phone }}" class="block p-3 mb-2 rounded border {% if active_phone == s.phone %}bg-blue-50 border-blue-400{% else %}bg-gray-50 hover:bg-gray-100 border-gray-200{% endif %} transition text-xs">
                    <div class="font-bold text-gray-800">{{ s.name }}</div>
                    <div class="text-gray-500 mt-0.5">رقم: {{ s.phone }}</div>
                </a>
            {% endfor %}
        </div>

        <!-- ساحة المحادثة -->
        <div class="bg-white rounded-lg shadow col-span-2 flex flex-col max-h-[78vh]">
            {% if active_session %}
                <div class="p-3 border-b bg-gray-50 flex justify-between items-center rounded-t-lg">
                    <div class="text-xs">
                        <h3 class="font-bold text-gray-800">{{ active_session.name }}</h3>
                        <p class="text-gray-500">واتساب: {{ active_session.phone }}</p>
                    </div>
                    <form action="/dashboard/close" method="post">
                        <input type="hidden" name="phone" value="{{ active_session.phone }}">
                        <button type="submit" class="bg-red-500 text-white px-3 py-1 rounded text-xs hover:bg-red-600 transition">إنهاء المحادثة وحذفها</button>
                    </form>
                </div>

                <div class="flex-1 p-3 overflow-y-auto space-y-3 bg-gray-50/50">
                    {% for msg in active_session.history %}
                        {% if msg.role == 'user' %}
                            <div class="flex justify-start">
                                <div class="bg-white text-gray-800 rounded-lg p-3 max-w-sm shadow-sm border text-xs">
                                    <span class="block text-[10px] font-bold text-blue-600 mb-0.5">العميل</span>
                                    {{ msg.content }}
                                </div>
                            </div>
                        {% else %}
                            <div class="flex justify-end">
                                <div class="bg-blue-600 text-white rounded-lg p-3 max-w-sm shadow-sm text-xs">
                                    <span class="block text-[10px] font-bold text-blue-200 mb-0.5">النظام / أنت</span>
                                    {{ msg.content | replace("[DONE]", "") }}
                                </div>
                            </div>
                        {% endif %}
                    {% endfor %}
                </div>

                <div class="p-3 border-t bg-white rounded-b-lg">
                    <form action="/dashboard/reply" method="post" class="flex gap-2">
                        <input type="hidden" name="phone" value="{{ active_session.phone }}">
                        <input type="text" name="message" required placeholder="اكتب ردك المباشر للعميل هنا..." class="flex-1 border rounded px-3 py-1.5 text-xs focus:outline-none focus:ring-1 focus:ring-blue-500">
                        <button type="submit" class="bg-blue-600 text-white px-4 py-1.5 rounded text-xs hover:bg-blue-700 font-bold transition">إرسال</button>
                    </form>
                </div>
            {% else %}
                <div class="flex-1 flex flex-col items-center justify-center text-gray-400 p-6 text-sm">
                    <p>يرجى تحديد عميل من القائمة للبدء بالرد عليه ومتابعة تفاصيل عمله.</p>
                </div>
            {% endif %}
        </div>
    </main>
</body>
</html>
"""

@app.get("/dashboard", response_class=HTMLResponse)
def get_dashboard(request: Request, active: str = None):
    sessions = get_all_agent_sessions()
    active_session = None
    if active:
        for s in sessions:
            if s["phone"] == active:
                active_session = s
                break
    from jinja2 import Template
    return Template(DASHBOARD_HTML).render(sessions=sessions, active_phone=active, active_session=active_session)

@app.post("/dashboard/reply")
def agent_reply(phone: str = Form(...), message: str = Form(...)):
    wa_send_text(phone, message)
    history = load_session(phone)
    history.append({"role": "assistant", "content": message})
    save_session(phone, history, status="agent")
    return RedirectResponse(url=f"/dashboard?active={phone}", status_code=303)

@app.post("/dashboard/close")
def agent_close_session(phone: str = Form(...)):
    delete_session(phone)
    return RedirectResponse(url="/dashboard", status_code=303)
