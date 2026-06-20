import os
import json
import requests
from flask import Flask, request, jsonify
from openai import OpenAI

app = Flask(__name__)

PHONE_NUMBER_ID = os.environ.get("WHATSAPP_PHONE_NUMBER_ID")
ACCESS_TOKEN = os.environ.get("WHATSAPP_ACCESS_TOKEN")
VERIFY_TOKEN = os.environ.get("WHATSAPP_VERIFY_TOKEN")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

WHATSAPP_API_URL = f"https://graph.facebook.com/v21.0/{PHONE_NUMBER_ID}/messages"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

sessions: dict[str, list] = {}

SYSTEM_PROMPT = """الدور: أنت مساعد ذكي ومحترف لخدمة العملاء.
سياسة اللغة: تواصل مع العميل باللغة العربية الفصحى فقط.

مهمتك: جمع 3 معلومات من المستخدم خطوة بخطوة (معلومة واحدة في كل رسالة):
1. الاسم الثلاثي
2. رقم الهوية
3. رقم الجوال

قواعد مهمة:
- لا تطلب أكثر من معلومة واحدة في نفس الرسالة
- إذا أرسل المستخدم زر (تتبع الطلب أو الدعم الفني)، ابدأ فوراً بطلب الاسم الثلاثي
- كن مهذباً ومحترفاً في جميع الأوقات

بعد جمع الاسم الثلاثي ورقم الهوية ورقم الجوال بنجاح، أرسل هذه الرسالة:
"شكراً لك يا [الاسم الثلاثي]. صاحب الهوية ([رقم الهوية]) لقد تم استلام بياناتك بنجاح. نحن نقدر تعاونك معنا. سوف يتم تحويلك للموظف بأسرع وقت."

ثم أضف في نهاية ردك الكلمة السحرية: [DONE]"""


def send_text(to: str, text: str) -> None:
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": text},
    }
    requests.post(WHATSAPP_API_URL, json=payload, headers=headers)


def send_menu_buttons(to: str) -> None:
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "interactive",
        "interactive": {
            "type": "button",
            "body": {"text": "مرحباً بك! 👋\nكيف يمكنني مساعدتك اليوم؟"},
            "action": {
                "buttons": [
                    {
                        "type": "reply",
                        "reply": {"id": "order_tracking", "title": "📦 تتبع الطلب"},
                    },
                    {
                        "type": "reply",
                        "reply": {"id": "technical_support", "title": "🛠 الدعم الفني"},
                    },
                ]
            },
        },
    }
    requests.post(WHATSAPP_API_URL, json=payload, headers=headers)


def ask_ai(phone: str, user_input: str) -> str:
    sessions[phone].append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="openai/o4-mini",
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + sessions[phone],
    )

    reply = response.choices[0].message.content or ""
    sessions[phone].append({"role": "assistant", "content": reply})
    return reply


@app.route("/webhook", methods=["GET"])
def verify():
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        print("Webhook verified")
        return challenge, 200
    return "Forbidden", 403


@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.json or {}
    print(f"DEBUG_RECEIVED: {json.dumps(data)}")

    if data.get("object") != "whatsapp_business_account":
        return jsonify({"status": "ignored"}), 200

    try:
        value = data["entry"][0]["changes"][0]["value"]
        messages = value.get("messages")

        if not messages:
            return jsonify({"status": "no_messages"}), 200

        message = messages[0]
        from_number: str = message["from"]
        display_number: str = value["metadata"]["display_phone_number"]

        # تجاهل رسائل البوت لنفسه
        if from_number == display_number:
            return jsonify({"status": "self"}), 200

        msg_type = message.get("type")

        # استخراج نص الرسالة
        if msg_type == "text":
            user_input = message["text"]["body"].strip()
        elif msg_type == "interactive":
            user_input = message["interactive"]["button_reply"]["title"]
        else:
            return jsonify({"status": "unsupported"}), 200

        # مستخدم جديد → أرسل الأزرار وابدأ session
        if from_number not in sessions:
            sessions[from_number] = []
            send_menu_buttons(from_number)
            return jsonify({"status": "menu_sent"}), 200

        # مستخدم موجود → أرسل للـ AI
        reply = ask_ai(from_number, user_input)

        if "[DONE]" in reply:
            clean_reply = reply.replace("[DONE]", "").strip()
            send_text(from_number, clean_reply)
            del sessions[from_number]
            print(f"Session cleared for {from_number}")
        else:
            send_text(from_number, reply)

    except Exception as e:
        print(f"Error: {e}")

    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting bot on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
