"""
"Мозг" ассистента: правила, инструмент записи, общая логика.
Не зависит ни от Telegram, ни от конкретного ИИ — провайдер подключается в backends.py.
"""
import json
import logging
import os
from datetime import datetime
from zoneinfo import ZoneInfo

from salon import salon_info_text

TZ = ZoneInfo("Asia/Almaty")
MAX_HISTORY = 20  # сколько последних сообщений помнить на клиента

BOOKING_NAME = "create_booking"
BOOKING_DESCRIPTION = (
    "Создать, перенести или отменить запись клиента. action='new' — новая запись, "
    "'reschedule' — перенос уже существующей записи, 'cancel' — отмена. "
    "Вызывай ТОЛЬКО после того, как повторил клиенту все детали и он ответил согласием отдельным сообщением."
)
BOOKING_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["new", "reschedule", "cancel"],
                   "description": "new — новая запись, reschedule — перенос, cancel — отмена"},
        "client_name": {"type": "string", "description": "Имя клиента"},
        "phone": {"type": "string", "description": "Телефон клиента"},
        "service": {"type": "string", "description": "Услуга из прайса"},
        "datetime": {"type": "string", "description": "Дата и время записи (для переноса — НОВОЕ время), например '12 октября, 15:00'"},
        "comment": {"type": "string", "description": "Пожелания: мастер и т.п. Может быть пустым"},
    },
    "required": ["action", "client_name", "phone", "service", "datetime"],
}
FALLBACK = "Минуточку, уточню у администратора 🙏"


class BookingStore:
    """Последняя запись каждого клиента. Хранится в JSON-файле, чтобы переживать перезапуски."""

    def __init__(self, path: str | None):
        self.path = path
        self.data: dict[str, dict] = {}
        if path and os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception:
                logging.exception("Не удалось прочитать %s, начинаю с пустого списка", path)

    def get(self, user_id: int) -> dict | None:
        return self.data.get(str(user_id))

    def set(self, user_id: int, booking: dict | None):
        if booking is None:
            self.data.pop(str(user_id), None)
        else:
            self.data[str(user_id)] = booking
        if self.path:
            try:
                with open(self.path, "w", encoding="utf-8") as f:
                    json.dump(self.data, f, ensure_ascii=False, indent=1)
            except Exception:
                logging.exception("Не удалось сохранить %s", self.path)


def describe(b: dict) -> str:
    return f"{b.get('service', '?')}, {b.get('datetime', '?')}, {b.get('client_name', '?')}, {b.get('phone', '?')}"


def system_prompt(active: dict | None = None) -> str:
    now = datetime.now(TZ).strftime("%A, %d.%m.%Y, %H:%M")
    active_text = f"У этого клиента уже есть заявка: {describe(active)}." if active \
        else "У этого клиента нет активных заявок."
    return f"""Ты — вежливый администратор салона в мессенджере. Сейчас в Алматы: {now}.

ИНФОРМАЦИЯ О САЛОНЕ (единственный источник фактов):
{salon_info_text()}

ЗАПИСИ КЛИЕНТА: {active_text}

ПРАВИЛА:
1. Отвечай на языке клиента (русский, казахский или английский). Коротко, тепло, как живой администратор, 1–4 предложения.
2. Цены, услуги, адрес и часы бери ТОЛЬКО из информации выше. Ничего не выдумывай. Если услуги нет в списке — НЕ говори, что салон её не делает: скажи, что в прайсе её нет и ты уточнишь у администратора, и можешь предложить похожую услугу из списка.
3. Никогда не угадывай детали записи клиента (услугу, дату, время). Если чего-то не знаешь — спроси.
4. Новая запись: узнай услугу, дату и время, имя и телефон. Время — только в часы работы салона.
5. Перенос или отмена: если у клиента есть заявка (см. «ЗАПИСИ КЛИЕНТА»), используй её данные и не спрашивай их заново — уточни только новое время. Если заявки нет — спроси, на какую услугу и время он был записан.
6. ОБЯЗАТЕЛЬНО перед вызовом {BOOKING_NAME}: отдельным сообщением повтори детали одной строкой и спроси «Всё верно?». Для переноса покажи «было → стало». Вызывай {BOOKING_NAME} ТОЛЬКО после согласия клиента в СЛЕДУЮЩЕМ сообщении. Если клиент ответил неоднозначно — переспроси.
7. Ты не подтверждаешь, что время точно свободно: говори, что администратор подтвердит в ближайшее время.
8. Не давай медицинских советов (аллергия, кожа, беременность) — предлагай обсудить с мастером.
9. На посторонние темы мягко возвращай разговор к салону.
"""


class Assistant:
    def __init__(self, backend, notify, store: BookingStore | None = None):
        """
        backend — объект из backends.py (Gemini или Claude)
        notify  — async-функция(text), которая отправит заявку владельцу
        store   — где хранить последние записи клиентов
        """
        self.backend = backend
        self.notify = notify
        self.store = store or BookingStore(None)

    def reset(self, user_id: int):
        self.backend.reset(user_id)

    async def reply(self, user_id: int, text: str, username: str = "") -> str:
        async def on_booking(args: dict) -> str:
            action = args.get("action", "new")
            previous = self.store.get(user_id)
            await self.notify(format_booking(args, username, previous))
            self.store.set(user_id, None if action == "cancel" else
                           {k: args.get(k) for k in ("service", "datetime", "client_name", "phone")})
            if action == "cancel":
                return "Отмена отправлена администратору. Подтверди клиенту, что запись отменена."
            return "Заявка отправлена администратору. Скажи клиенту, что с ним свяжутся для подтверждения."

        system = system_prompt(self.store.get(user_id))
        answer = await self.backend.reply(user_id, text, system, on_booking)
        return answer.strip() or FALLBACK


TITLES = {
    "new": "🆕 Новая заявка на запись",
    "reschedule": "🔁 ПЕРЕНОС записи",
    "cancel": "❌ ОТМЕНА записи",
}


def format_booking(data: dict, username: str = "", previous: dict | None = None) -> str:
    action = data.get("action", "new")
    lines = [
        TITLES.get(action, TITLES["new"]),
        f"Имя: {data.get('client_name', '-')}",
        f"Телефон: {data.get('phone', '-')}",
        f"Услуга: {data.get('service', '-')}",
    ]
    if action == "reschedule":
        if previous:
            lines.append(f"Было: {previous.get('datetime', '-')}")
        lines.append(f"Стало: {data.get('datetime', '-')}")
    else:
        lines.append(f"Когда: {data.get('datetime', '-')}")
    if data.get("comment"):
        lines.append(f"Комментарий: {data['comment']}")
    if username:
        lines.append(f"Telegram: @{username}")
    return "\n".join(lines)
