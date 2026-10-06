"""
"Мозг" ассистента: правила, инструмент записи, общая логика.
Не зависит ни от Telegram, ни от конкретного ИИ — провайдер подключается в backends.py.
"""
from datetime import datetime
from zoneinfo import ZoneInfo

from salon import salon_info_text

TZ = ZoneInfo("Asia/Almaty")
MAX_HISTORY = 20  # сколько последних сообщений помнить на клиента

BOOKING_NAME = "create_booking"
BOOKING_DESCRIPTION = (
    "Записать клиента на услугу. Вызывай ТОЛЬКО после того, как ты повторил клиенту "
    "все детали (услуга, время, имя, телефон) и он ответил согласием отдельным сообщением."
)
BOOKING_SCHEMA = {
    "type": "object",
    "properties": {
        "client_name": {"type": "string", "description": "Имя клиента"},
        "phone": {"type": "string", "description": "Телефон клиента"},
        "service": {"type": "string", "description": "Услуга из прайса"},
        "datetime": {"type": "string", "description": "Желаемая дата и время, например '12 октября, 15:00'"},
        "comment": {"type": "string", "description": "Пожелания: мастер и т.п. Может быть пустым"},
    },
    "required": ["client_name", "phone", "service", "datetime"],
}
BOOKING_DONE = "Заявка отправлена администратору. Скажи клиенту, что с ним свяжутся для подтверждения."
FALLBACK = "Минуточку, уточню у администратора 🙏"


def system_prompt() -> str:
    now = datetime.now(TZ).strftime("%A, %d.%m.%Y, %H:%M")
    return f"""Ты — вежливый администратор салона в мессенджере. Сейчас в Алматы: {now}.

ИНФОРМАЦИЯ О САЛОНЕ (единственный источник фактов):
{salon_info_text()}

ПРАВИЛА:
1. Отвечай на языке клиента (русский, казахский или английский). Коротко, тепло, как живой администратор, 1–4 предложения.
2. Цены, услуги, адрес и часы бери ТОЛЬКО из информации выше. Ничего не выдумывай. Если услуги нет в списке — НЕ говори, что салон её не делает: скажи, что в прайсе её нет и ты уточнишь у администратора, и можешь предложить похожую услугу из списка.
3. Чтобы записать: узнай услугу, удобные дату и время, имя и телефон. Время — только в часы работы салона.
4. ОБЯЗАТЕЛЬНО перед записью: когда все данные собраны, отдельным сообщением повтори их одной строкой (услуга, дата и время, имя, телефон) и спроси «Всё верно?». Вызывай {BOOKING_NAME} ТОЛЬКО после того, как клиент ответил согласием в СЛЕДУЮЩЕМ сообщении. Никогда не вызывай {BOOKING_NAME} в том же ответе, где впервые получил телефон.
5. Ты не подтверждаешь, что время точно свободно: говори, что администратор подтвердит запись в ближайшее время.
6. Не давай медицинских советов (аллергия, кожа, беременность) — предлагай обсудить с мастером.
7. На посторонние темы мягко возвращай разговор к салону.
"""


class Assistant:
    def __init__(self, backend, notify):
        """
        backend — объект из backends.py (Gemini или Claude)
        notify  — async-функция(text), которая отправит заявку владельцу
        """
        self.backend = backend
        self.notify = notify

    def reset(self, user_id: int):
        self.backend.reset(user_id)

    async def reply(self, user_id: int, text: str, username: str = "") -> str:
        async def on_booking(args: dict) -> str:
            await self.notify(format_booking(args, username))
            return BOOKING_DONE

        answer = await self.backend.reply(user_id, text, system_prompt(), on_booking)
        return answer.strip() or FALLBACK


def format_booking(data: dict, username: str = "") -> str:
    lines = [
        "🆕 Новая заявка на запись",
        f"Имя: {data.get('client_name', '-')}",
        f"Телефон: {data.get('phone', '-')}",
        f"Услуга: {data.get('service', '-')}",
        f"Когда: {data.get('datetime', '-')}",
    ]
    if data.get("comment"):
        lines.append(f"Комментарий: {data['comment']}")
    if username:
        lines.append(f"Telegram: @{username}")
    return "\n".join(lines)
