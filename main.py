"""
WhatsApp AI Chatbot - FastAPI + LangGraph + OpenRouter + Mini Dashboard
سريع، موثوق، ويدعم الرد البشري ولوحة تحكم مصغرة
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
from fastapi.templating import Jinja2Templates
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

# لوحة التحكم المحمية برمز بسيط (اختياري، يمكنك تعيينه في ريلوي)
DASHBOARD_PASSWORD = os.environ.get("DASHBOARD_PASSWORD", "admin123")

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

ai = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_KEY,
)

# ─── SQLite Sessions المحدثة لدعم الحالات ──────────────────────────────────────────
# الحالات المتاحة: 'bot' (البوت شغال) ، 'agent' (مع الموظف البشري)
def init_db() -> None:
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            phone TEXT PRIMARY KEY,
            history TEXT NOT NULL DEFAULT '[]',
            status TEXT NOT NULL DEFAULT 'bot',
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    con.commit()
    con.close()
    log.info("Database initialized at %s", DB_PATH)

def load_session(phone: str) -> tuple[list[dict], str]:
    con = sqlite3.connect(DB_PATH)
    row = con.execute("SELECT history, status FROM sessions WHERE phone=?", (phone,)).fetchone()
    con.close()
    if row:
        return json.loads(row[0]), row[1]
    return [], "bot"

def save_session(phone: str, history: list[dict], status: str = "bot") -> None:
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT INTO sessions (phone, history, status, updated_at) VALUES (?, ?, ?, CURRENT_TIMESTAMP) "
        "ON CONFLICT(phone) DO UPDATE SET history=excluded.history, status=excluded.status, updated_at=CURRENT_TIMESTAMP",
        (phone, json.dumps(history, ensure_ascii=False), status),
    )
    con.commit()
    con.close()

def mark_as_agent(phone: str) -> None:
    con = sqlite3.connect(DB_PATH)
    con.execute("UPDATE sessions SET status='agent', updated_at=CURRENT_TIMESTAMP WHERE phone=?", (phone,))
    con.commit()
    con.close()
    log.info("Session %s transferred to Agent", phone)

def close_session(phone: str) -> None:
    # إعادة تصفير الجلسة أو حذفها عند إنهاء الموظف للمحادثة
    con = sqlite3.connect(DB_PATH)
    con.execute("DELETE FROM sessions WHERE phone=?", (phone,))
    con.commit()
    con.close()
    log.info("Session closed and cleared for %s", phone)

def get_all_agent_sessions() -> list[dict]:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    rows = con.execute("SELECT * FROM sessions WHERE status='agent' ORDER BY updated_at DESC").fetchall()
    con.close()
    
    result = []
    for r in rows:
        history = json.loads(r["history"])
        # محاولة استخراج الاسم من آخر رسائل لتسهيل العرض في اللوحة
        name = r["phone"]
        for msg in reversed(history):
            if msg["role"] == "assistant" and "شكراً لك يا" in msg["content"]:
                match = re.search(r"شكراً لك يا (.+?)\.", msg["content"])
                if match:
                    name = match.group(1).strip()
                    break
        result.append({
            "phone": r["phone"],
            "name": name,
            "history": history,
            "updated_at": r["updated_at"]
        })
    return result

# ─── Per-Phone Locks ──────────────────────────────────────────────────────────
_phone_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
_locks_guard = threading.Lock()

def get_phone_lock(phone: str) -> threading.Lock:
    with _locks_guard:
        return _phone_locks[phone]

# ─── Google Sheets + Email ────────────────────────────────────────────────────
_sheets_client = None
def get_sheet():
    global _sheets_client
    if not GOOGLE_SHEET_ID or not GOOGLE_SERVICE_ACCOUNT_JSON:
        return None
    try:
        if _sheets_client is None:
            creds_info = json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)
            creds = Credentials.from_service_account_info(creds_info, scopes=["https://www.googleapis.com/auth/spreadsheets"])
            _sheets_client = gspread.authorize(creds)
        return _sheets_client.open_by_key(GOOGLE_SHEET_ID).sheet1
    except Exception as e:
        log.error("Google Sheets connection error: %s", e)
        return None

def log_to_sheet(whatsapp_phone: str, name: str, national_id: str, contact_phone: str) -> None:
    sheet = get_sheet()
    if sheet is None: return
    try:
        sheet.append_row([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), whatsapp_phone, name, national_id, contact_phone])
    except Exception as e: log.error("Google Sheets insert error: %s", e)

def send_notification_email(whatsapp_phone: str, name: str, national_id: str, contact_phone: str) -> None:
    if not (RESEND_API_KEY and NOTIFY_EMAIL): return
    body_html = f"<p>عميل جديد ينتظر التواصل:</p><p><b>الاسم:</b> {name}<br><b>رقم الهوية:</b> {national_id}<br><b>رقم الجوال:</b> {contact_phone}<br><b>واتساب:</b> {whatsapp_phone}</p>"
    try:
        requests.post("https://api.resend.com/emails", headers={"Authorization": f"Bearer {RESEND_API_KEY}", "Content-Type": "application/json"},
                      json={"from": NOTIFY_FROM_EMAIL, "to": [NOTIFY_EMAIL], "subject": f"عميل جديد يحتاج للتواصل - {name}", "html": body_html}, timeout=10)
    except Exception as e: log.error("Email notification error: %s", e)

def extract_customer_data(final_message: str, last_user_input: str) -> dict | None:
    match = re.search(r"شكراً لك يا (.+?)\.\s*صاحب الهوية \((.+?)\)", final_message)
    if not match: return None
    return {"name": match.group(1).strip(), "national_id": match.group(2).strip(), "contact_phone": last_user_input.strip()}

# ─── WhatsApp API ─────────────────────────────────────────────────────────────
def _wa_headers() -> dict:
    return {"Authorization": f"Bearer {ACCESS_TOKEN}", "Content-Type": "application/json"}

def wa_send_text(to: str, text: str) -> None:
    payload = {"messaging_product": "whatsapp", "to": to, "type": "text", "text": {"body": text}}
    r = requests.post(WA_URL, json=payload, headers=_wa_headers(), timeout=10)
    if not r.ok: log.error("WhatsApp error: %s", r.text)

def wa_send_buttons(to: str) -> None:
    payload = {
        "messaging_product": "whatsapp", "to": to, "type": "interactive",
        "interactive": {
            "type": "button", "body": {"text": "مرحباً بك! 👋\nكيف يمكنني مساعدتك اليوم؟"},
            "action": {"buttons": [{"type": "reply", "reply": {"id": "order_tracking", "title": "📦 تتبع الطلب"}},
                                   {"type": "reply", "reply": {"id": "technical_support", "title": "🛠 الدعم الفني"}}]}
        }
    }
    r = requests.post(WA_URL, json=payload, headers=_wa_headers(), timeout=10)
    if not r.ok: log.error("WhatsApp buttons error: %s", r.text)

# ─── LangGraph State ──────────────────────────────────────────────────────────
class BotState(TypedDict):
    messages: Annotated[list, add_messages]
    phone: str
    done: bool

def ai_node(state: BotState) -> BotState:
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage): history.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage): history.append({"role": "assistant", "content": msg.content})

    response = ai.chat.completions.create(model=MODEL, messages=history)
    reply = response.choices[0].message.content or ""
    done = "[DONE]" in reply
    clean = reply.replace("[DONE]", "").strip()

    wa_send_text(state["phone"], clean)
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
        history, status = load_session(phone)

        # إذا كانت المحادثة مع الموظف حالياً، نقوم فقط بحفظ رسالة العميل وتحديث الوقت ليراها الموظف في اللوحة
        if status == "agent":
            history.append({"role": "user", "content": user_input})
            save_session(phone, history, status="agent")
            log.info("User %s sent a message to Agent chat.", phone)
            return

        if not history:
            wa_send_buttons(phone)
            welcome_msg = "مرحباً بك! 👋 كيف يمكنني مساعدتك اليوم؟"
            save_session(phone, [{"role": "assistant", "content": welcome_msg}], status="bot")
            return

        messages = []
        for item in history:
            if item["role"] == "user": messages.append(HumanMessage(content=item["content"]))
            else: messages.append(AIMessage(content=item["content"]))
        messages.append(HumanMessage(content=user_input))

        state = graph.invoke({"messages": messages, "phone": phone, "done": False})

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
            
            # بدلاً من المسح الكامل، نقوم بتحويل الحالة لـ agent وتثبيت المحادثة
            save_session(phone, new_history, status="agent")
            log.info("Session %s marked as AGENT ready", phone)
        else:
            save_session(phone, new_history, status="bot")

# ─── FastAPI App ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    log.info("Bot started | Model: %s", MODEL)
    yield

app = FastAPI(title="WhatsApp AI Bot & Dashboard", lifespan=lifespan)

# HTML Template Embedded Directly (لعدم الحاجة لإنشاء ملفات إضافية)
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <title>لوحة تحكم الدعم الفني المصغرة</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&display=swap" rel="stylesheet">
    <style> body { font-family: 'Cairo', sans-serif; } </style>
    <meta http-equiv="refresh" content="15"> </head>
<body class="bg-gray-100 min-h-screen flex flex-col">
    <header class="bg-indigo-600 text-white p-4 shadow-md flex justify-between items-center">
        <h1 class="text-xl font-bold">💬 لوحة إدارة محادثات واتساب (الرد البشري)</h1>
        <span class="bg-indigo-700 px-3 py-1 rounded text-sm">المحادثات النشطة المنتظرة</span>
    </header>

    <main class="flex-1 p-6 max-w-6xl w-full mx-auto grid grid-cols-1 md:grid-cols-3 gap-6">
        <div class="bg-white rounded-lg shadow p-4 col-span-1 overflow-y-auto max-h-[75vh]">
            <h2 class="font-bold text-gray-700 mb-4 border-b pb-2">📦 عملاء في الانتظار</h2>
            {% if not sessions %}
                <p class="text-gray-400 text-center py-8 text-sm">لا توجد محادثات محولة حالياً.</p>
            {% endif %}
            {% for s in sessions %}
                <a href="?active={{ s.phone }}" class="block p-3 mb-2 rounded border {% if active_phone == s.phone %}bg-indigo-50 border-indigo-500{% else %}bg-gray-50 hover:bg-gray-100 border-gray-200{% endif %} transition">
                    <div class="font-bold text-gray-800 text-sm">{{ s.name }}</div>
                    <div class="text-xs text-gray-500 mt-1">رقم: {{ s.phone }}</div>
                    <div class="text-[10px] text-gray-400 text-left mt-1">{{ s.updated_at }}</div>
                </a>
            {% endfor %}
        </div>

        <div class="bg-white rounded-lg shadow col-span-2 flex flex-col max-h-[75vh]">
            {% if active_session %}
                <div class="p-4 border-b bg-gray-50 flex justify-between items-center rounded-t-lg">
                    <div>
                        <h3 class="font-bold text-gray-800">{{ active_session.name }}</h3>
                        <p class="text-xs text-gray-500">واتساب: {{ active_session.phone }}</p>
                    </div>
                    <form action="/dashboard/close" method="post">
                        <input type="hidden" name="phone" value="{{ active_session.phone }}">
                        <button type="submit" class="bg-red-500 text-white px-3 py-1 rounded text-xs hover:bg-red-600 transition">إنهاء المحادثة (إغلاق)</button>
                    </form>
                </div>

                <div class="flex-1 p-4 overflow-y-auto space-y-3 bg-gray-50/50">
                    {% for msg in active_session.history %}
                        {% if msg.role == 'user' %}
                            <div class="flex justify-start">
                                <div class="bg-white text-gray-800 rounded-lg p-3 max-w-md shadow-sm border border-gray-100 text-sm">
                                    <span class="block text-[10px] font-bold text-indigo-600 mb-1">العميل</span>
                                    {{ msg.content }}
                                </div>
                            </div>
                        {% else %}
                            <div class="flex justify-end">
                                <div class="bg-indigo-600 text-white rounded-lg p-3 max-w-md shadow-sm text-sm">
                                    <span class="block text-[10px] font-bold text-indigo-200 mb-1">💡 النظام/الموظف</span>
                                    {{ msg.content | replace("[DONE]", "") }}
                                </div>
                            </div>
                        {% endif %}
                    {% endfor %}
                </div>

                <div class="p-4 border-t bg-white rounded-b-lg">
                    <form action="/dashboard/reply" method="post" class="flex gap-2">
                        <input type="hidden" name="phone" value="{{ active_session.phone }}">
                        <input type="text" name="message" required placeholder="اكتب ردك هنا وسيتم إرساله مباشرة للعميل عبر واتساب..." class="flex-1 border rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500">
                        <button type="submit" class="bg-indigo-600 text-white px-5 py-2 rounded text-sm hover:bg-indigo-700 font-bold transition">إرسال 🚀</button>
                    </form>
                </div>
            {% else %}
                <div class="flex-1 flex flex-col items-center justify-center text-gray-400 p-8">
                    <span class="text-5xl mb-2">👈</span>
                    <p>يرجى اختيار محادثة من القائمة الجانبية لبدء المتابعة والرد.</p>
                </div>
            {% endif %}
        </div>
    </main>
</body>
</html>
"""

@app.get("/dashboard", response_class=HTMLResponse)
def get_dashboard(request: Request, active: str = None):
    # جلب كافة المحادثات التي تم تحويلها للعميل البشري
    sessions = get_all_agent_sessions()
    active_session = None
    if active:
        for s in sessions:
            if s["phone"] == active:
                active_session = s
                break
                
    # رندرة الصفحة مباشرة باستخدام الـ string المخزن كـ Template
    from jinja2 import Template
    tmpl = Template(DASHBOARD_HTML)
    return tmpl.render(sessions=sessions, active_phone=active, active_session=active_session)

@app.post("/dashboard/reply")
def agent_reply(phone: str = Form(...), message: str = Form(...)):
    # 1. إرسال الرسالة للعميل عبر واتساب فوراً
    wa_send_text(phone, message)
    
    # 2. تحديث سجل المحادثة داخل الـ SQLite بقيمة الموظف
    history, status = load_session(phone)
    history.append({"role": "assistant", "content": message})
    save_session(phone, history, status="agent")
    
    return RedirectResponse(url=f"/dashboard?active={phone}", status_code=303)

@app.post("/dashboard/close")
def agent_close_session(phone: str = Form(...)):
    # إغلاق الجلسة ومسحها حتى يتمكن البوت من العمل مجدداً إذا أرسل العميل مستقبلاً رسالة جديدة
    close_session(phone)
    return RedirectResponse(url="/dashboard", status_code=303)

# ─── Webhook Endpoints ────────────────────────────────────────────────────────
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
    if data.get("object") != "whatsapp_business_account"):
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
        if msg_type == "text":
            user_input = message["text"]["body"].strip()
        elif msg_type == "interactive":
            user_input = message["interactive"]["button_reply"]["title"]
        else:
            return JSONResponse({"status": "unsupported"})

        background.add_task(handle_message, from_number, user_input)

    except (KeyError, IndexError) as e:
        log.warning("Parse error: %s", e)

    return JSONResponse({"status": "ok"})

@app.get("/health")
def health():
    return {"status": "running", "model": MODEL}
