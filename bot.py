"""
Telegram-бот: ИИ-администратор салона.
Запуск: python bot.py  (переменные окружения — см. .env.example)
"""
import asyncio
import logging
import os

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, CommandStart
from aiogram.types import Message
from dotenv import load_dotenv

from assistant import Assistant, BookingStore
from backends import make_backend
from salon import SALON

load_dotenv()
logging.basicConfig(level=logging.INFO)

BOT_TOKEN = os.environ["BOT_TOKEN"]
OWNER_CHAT_ID = int(os.environ["OWNER_CHAT_ID"])
backend = make_backend(
    provider=os.getenv("PROVIDER", "gemini"),
    gemini_key=os.getenv("GEMINI_API_KEY", ""),
    claude_key=os.getenv("ANTHROPIC_API_KEY", ""),
    model=os.getenv("MODEL", ""),
)

bot = Bot(BOT_TOKEN)
dp = Dispatcher()


async def notify_owner(text: str):
    await bot.send_message(OWNER_CHAT_ID, text)


assistant = Assistant(backend, notify_owner, BookingStore(os.getenv("BOOKINGS_FILE", "bookings.json")))
locks: dict[int, asyncio.Lock] = {}


@dp.message(CommandStart())
async def start(message: Message):
    assistant.reset(message.from_user.id)
    await message.answer(
        f"Здравствуйте! Я онлайн-администратор {SALON['name']} 💇‍♀️\n"
        "Подскажу цены, расскажу об услугах и запишу вас. Чем могу помочь?"
    )


@dp.message(Command("id"))
async def my_id(message: Message):
    # Нужна один раз: узнать свой chat id для OWNER_CHAT_ID
    await message.answer(f"Ваш chat id: {message.chat.id}")


@dp.message(F.text)
async def chat(message: Message):
    uid = message.from_user.id
    lock = locks.setdefault(uid, asyncio.Lock())
    async with lock:  # чтобы два быстрых сообщения не перепутали историю
        await bot.send_chat_action(message.chat.id, "typing")
        try:
            answer = await assistant.reply(uid, message.text, message.from_user.username or "")
        except Exception:
            logging.exception("AI error")
            assistant.reset(uid)
            answer = f"Извините, небольшая техническая заминка. Позвоните нам: {SALON['phone']}"
        await message.answer(answer)


@dp.message()
async def other(message: Message):
    await message.answer("Я понимаю только текстовые сообщения 🙏 Напишите, пожалуйста, вопрос текстом.")


async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
