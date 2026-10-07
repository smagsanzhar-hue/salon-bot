"""
Провайдеры ИИ. Логика одна, «мозг» можно менять одной переменной PROVIDER:
- gemini — бесплатный лимит Google, для демо и первых тестов
- claude — платный (Anthropic), для клиентов, когда бот должен работать стабильно
"""
import logging

from assistant import BOOKING_NAME, BOOKING_DESCRIPTION, BOOKING_SCHEMA, MAX_HISTORY

MAX_ROUNDS = 3  # ответ -> запись -> ответ


class GeminiBackend:
    def __init__(self, api_key: str, model: str, client=None, fallback_models=()):
        from google import genai
        from google.genai import errors, types
        self.types, self.errors = types, errors
        # Не больше 2 коротких попыток на модель, чтобы клиент не ждал по 30 секунд
        self.client = client or genai.Client(api_key=api_key, http_options=types.HttpOptions(
            timeout=20_000,
            retry_options=types.HttpRetryOptions(attempts=2, initial_delay=1, max_delay=3),
        ))
        # Основная модель + запасные: если одна перегружена (503) или упёрлась в лимит (429), пробуем следующую
        self.models = [model, *[m for m in fallback_models if m and m != model]]
        self.history: dict[int, list] = {}
        self.tool = types.Tool(function_declarations=[types.FunctionDeclaration(
            name=BOOKING_NAME, description=BOOKING_DESCRIPTION,
            parameters_json_schema=BOOKING_SCHEMA,
        )])

    def reset(self, user_id: int):
        self.history.pop(user_id, None)

    async def _generate(self, contents, config):
        last_error = None
        for model in self.models:
            try:
                return await self.client.aio.models.generate_content(
                    model=model, contents=contents, config=config)
            except self.errors.ServerError as err:              # 5xx: перегрузка у Google
                last_error = err
            except self.errors.ClientError as err:              # 429: лимит; остальные 4xx — сразу ошибка
                if getattr(err, "code", None) != 429:
                    raise
                last_error = err
            logging.warning("Gemini %s недоступна (%s), пробую следующую модель", model, last_error)
        raise last_error

    async def reply(self, user_id, text, system, on_booking) -> str:
        msgs = self.history.setdefault(user_id, [])
        start = len(msgs)
        try:
            return await self._reply(user_id, msgs, text, system, on_booking)
        except BaseException:
            # Сбой или таймаут: убираем только неудавшееся сообщение, остальной разговор сохраняем
            del msgs[start:]
            raise

    async def _reply(self, user_id, msgs, text, system, on_booking) -> str:
        t = self.types
        msgs.append(t.Content(role="user", parts=[t.Part.from_text(text=text)]))
        config = t.GenerateContentConfig(
            system_instruction=system,
            tools=[self.tool],
            automatic_function_calling=t.AutomaticFunctionCallingConfig(disable=True),
        )
        answer = ""
        for _ in range(MAX_ROUNDS):
            resp = await self._generate(list(msgs), config)
            if not resp.candidates or not resp.candidates[0].content:
                break
            content = resp.candidates[0].content
            msgs.append(content)  # сохраняем как есть — так требует Gemini для вызовов функций
            parts = content.parts or []
            answer = "".join(p.text for p in parts if getattr(p, "text", None) and not getattr(p, "thought", False))
            calls = [p.function_call for p in parts if getattr(p, "function_call", None)]
            if not calls:
                break
            results = []
            for fc in calls:
                if fc.name == BOOKING_NAME:
                    result = await on_booking(dict(fc.args or {}))
                    results.append(t.Part(function_response=t.FunctionResponse(
                        id=fc.id, name=fc.name, response={"result": result})))
            msgs.append(t.Content(role="user", parts=results))
        self.history[user_id] = self._trim(msgs)
        return answer

    @staticmethod
    def _trim(msgs):
        if len(msgs) <= MAX_HISTORY:
            return msgs
        cut = msgs[-MAX_HISTORY:]
        while cut and not (cut[0].role == "user" and any(getattr(p, "text", None) for p in cut[0].parts or [])):
            cut = cut[1:]
        return cut


class ClaudeBackend:
    def __init__(self, api_key: str, model: str, client=None):
        if client is None:
            from anthropic import AsyncAnthropic
            client = AsyncAnthropic(api_key=api_key)
        self.client = client
        self.model = model
        self.history: dict[int, list] = {}
        self.tool = {"name": BOOKING_NAME, "description": BOOKING_DESCRIPTION, "input_schema": BOOKING_SCHEMA}

    def reset(self, user_id: int):
        self.history.pop(user_id, None)

    async def reply(self, user_id, text, system, on_booking) -> str:
        msgs = self.history.setdefault(user_id, [])
        start = len(msgs)
        try:
            return await self._reply(user_id, msgs, text, system, on_booking)
        except BaseException:
            del msgs[start:]  # сохраняем разговор, убираем только неудавшееся сообщение
            raise

    async def _reply(self, user_id, msgs, text, system, on_booking) -> str:
        msgs.append({"role": "user", "content": text})
        answer = ""
        for _ in range(MAX_ROUNDS):
            resp = await self.client.messages.create(
                model=self.model, max_tokens=600, system=system, tools=[self.tool], messages=msgs)
            msgs.append({"role": "assistant", "content": [_claude_param(b) for b in resp.content]})
            answer = "".join(b.text for b in resp.content if b.type == "text")
            if resp.stop_reason != "tool_use":
                break
            results = []
            for b in resp.content:
                if b.type == "tool_use" and b.name == BOOKING_NAME:
                    results.append({"type": "tool_result", "tool_use_id": b.id,
                                    "content": await on_booking(b.input)})
            msgs.append({"role": "user", "content": results})
        self.history[user_id] = self._trim(msgs)
        return answer

    @staticmethod
    def _trim(msgs):
        if len(msgs) <= MAX_HISTORY:
            return msgs
        cut = msgs[-MAX_HISTORY:]
        while cut and not (cut[0]["role"] == "user" and isinstance(cut[0]["content"], str)):
            cut = cut[1:]
        return cut


def _claude_param(block) -> dict:
    if block.type == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    return {"type": "text", "text": block.text}


DEFAULT_GEMINI_FALLBACKS = "gemini-3.1-flash-lite,gemini-3.5-flash"


def make_backend(provider: str, gemini_key: str = "", claude_key: str = "", model: str = "",
                 fallback_models: str = ""):
    provider = (provider or "gemini").lower()
    if provider == "gemini":
        fallbacks = [m.strip() for m in (fallback_models or DEFAULT_GEMINI_FALLBACKS).split(",")]
        return GeminiBackend(gemini_key, model or "gemini-3.5-flash-lite", fallback_models=fallbacks)
    if provider == "claude":
        return ClaudeBackend(claude_key, model or "claude-haiku-4-5-20251001")
    raise ValueError(f"Неизвестный PROVIDER: {provider} (нужно gemini или claude)")
