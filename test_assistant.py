"""Проверка логики без настоящего ИИ и без Telegram: python test_assistant.py"""
import asyncio
from types import SimpleNamespace as NS

from google.genai import types as gt

from assistant import Assistant, MAX_HISTORY
from backends import GeminiBackend, ClaudeBackend

BOOKING = {"client_name": "Айжан", "phone": "+77011234567",
           "service": "Маникюр с покрытием гель-лак", "datetime": "12 октября, 15:00"}


# ---------- Gemini: подменяем только сетевой вызов, типы настоящие ----------
def g_text(t):
    return gt.GenerateContentResponse(candidates=[gt.Candidate(
        content=gt.Content(role="model", parts=[gt.Part.from_text(text=t)]))])


def g_call(args):
    return gt.GenerateContentResponse(candidates=[gt.Candidate(content=gt.Content(role="model", parts=[
        gt.Part.from_text(text="Записываю!"),
        gt.Part(function_call=gt.FunctionCall(id="fc1", name="create_booking", args=args)),
    ]))])


class FakeGemini:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []
        self.aio = NS(models=self)

    async def generate_content(self, model, contents, config):
        self.calls.append({"contents": list(contents), "config": config})
        return self.responses.pop(0)


# ---------- Claude ----------
def c_text(t):
    return NS(stop_reason="end_turn", content=[NS(type="text", text=t)])


def c_call(args):
    return NS(stop_reason="tool_use", content=[
        NS(type="text", text="Записываю!"),
        NS(type="tool_use", id="tu1", name="create_booking", input=args)])


class FakeClaude:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []
        self.messages = self

    async def create(self, **kw):
        self.calls.append({**kw, "messages": list(kw["messages"])})
        return self.responses.pop(0)


def gemini(responses):
    fake = FakeGemini(responses)
    return GeminiBackend("x", "m", client=fake), fake


def claude(responses):
    fake = FakeClaude(responses)
    return ClaudeBackend("x", "m", client=fake), fake


CASES = [("Gemini", gemini, g_text, g_call), ("Claude", claude, c_text, c_call)]


async def check(name, make, text, call):
    sent = []
    async def notify(t): sent.append(t)

    # 1. обычный ответ, прайс в системном промпте
    be, fake = make([text("Маникюр стоит 7 000 ₸")])
    a = Assistant(be, notify)
    assert await a.reply(1, "Сколько стоит маникюр?") == "Маникюр стоит 7 000 ₸"
    sys = fake.calls[0]["config"].system_instruction if name == "Gemini" else fake.calls[0]["system"]
    assert "7 000" in sys and sent == []

    # 2. запись -> заявка владельцу -> финальный ответ
    be, fake = make([call(BOOKING), text("Готово, администратор скоро подтвердит запись.")])
    a = Assistant(be, notify)
    out = await a.reply(7, "Да, подтверждаю", username="aizhan")
    assert "подтвердит" in out
    assert len(sent) == 1 and "Айжан" in sent[0] and "@aizhan" in sent[0]
    second = fake.calls[1]["contents" if name == "Gemini" else "messages"]
    last = second[-1]
    if name == "Gemini":
        fr = last.parts[0].function_response
        assert fr.name == "create_booking" and fr.id == "fc1"
    else:
        assert last["content"][0]["type"] == "tool_result"

    # 3. у каждого клиента своя история
    be, _ = make([text("a"), text("b")])
    a = Assistant(be, notify)
    await a.reply(1, "привет"); await a.reply(2, "сәлем")
    assert len(be.history[1]) == 2 and len(be.history[2]) == 2

    # 4. пустой ответ ИИ -> вежливая заглушка
    be, _ = make([text("")])
    assert "уточню" in await Assistant(be, notify).reply(1, "?")

    # 5. длинный диалог обрезается и начинается с сообщения клиента
    be, _ = make([text(f"a{i}") for i in range(30)])
    a = Assistant(be, notify)
    for i in range(30):
        await a.reply(1, f"q{i}")
    h = be.history[1]
    assert len(h) <= MAX_HISTORY
    first_is_user_text = (h[0].role == "user" and h[0].parts[0].text) if name == "Gemini" \
        else (h[0]["role"] == "user" and isinstance(h[0]["content"], str))
    assert first_is_user_text
    print(f"ok: {name} — ответы, запись, история, обрезка")


async def main():
    for case in CASES:
        await check(*case)
    print("\nВсе проверки пройдены ✅")


asyncio.run(main())
