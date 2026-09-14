"""
Nutrition chatbot grounded in per-user stored progress.

Design choice worth calling out: we do NOT inject the user's full meal
history into the prompt on every turn. Instead the model gets tool
definitions (`get_daily_summary`, `get_meal_history`) and calls them when it
needs numbers, so context stays small and arithmetic (remaining protein,
etc.) happens in `storage.py`, not inside the model. This also means the
model can't silently drift into making up numbers for a day it hasn't
queried.

Scope guardrail: this bot is for logging/adherence/general nutrition
tips only. It should redirect medical/clinical questions (diagnosed
conditions, disordered-eating-adjacent requests, extreme calorie targets) to
a professional rather than answer them -- that's encoded in SYSTEM_PROMPT
below, not left to chance.
"""
from __future__ import annotations

import json
import os

from .storage import Storage

SYSTEM_PROMPT = """You are a nutrition tracking assistant for the NutriVision app.
You have tool access to the user's own logged meals and daily targets -- use
the tools rather than guessing or asking the user to repeat numbers they've
already logged.

Scope: meal logging questions, progress against daily targets, general
nutrition education, practical tips (meal timing, food swaps, hitting a
protein target, etc.).

Out of scope -- redirect to a doctor/dietitian instead of answering directly:
diagnosed medical or metabolic conditions, disordered-eating-adjacent
requests (extreme calorie restriction, compensatory behavior after eating),
supplement or medication interactions, and anything requiring a clinical
diagnosis. Be warm about the redirect, not clinical-sounding.
"""

TOOLS = [
    {
        "name": "get_daily_summary",
        "description": "Get the user's logged totals and remaining targets for a given date.",
        "input_schema": {
            "type": "object",
            "properties": {"date": {"type": "string", "description": "YYYY-MM-DD"}},
            "required": ["date"],
        },
    },
    {
        "name": "get_meal_history",
        "description": "Get the list of individual meals logged on a given date, with per-item breakdown.",
        "input_schema": {
            "type": "object",
            "properties": {"date": {"type": "string", "description": "YYYY-MM-DD"}},
            "required": ["date"],
        },
    },
]


class NutritionChatbot:
    def __init__(self, storage: Storage, user_id: str, model: str = "claude-sonnet-4-6"):
        self.storage = storage
        self.user_id = user_id
        self.model = model

    def _run_tool(self, name: str, tool_input: dict) -> dict:
        if name == "get_daily_summary":
            return self.storage.get_daily_summary(self.user_id, tool_input["date"])
        if name == "get_meal_history":
            return {"meals": self.storage.get_meals(self.user_id, tool_input["date"])}
        return {"error": f"unknown tool {name}"}

    def ask(self, user_message: str, conversation: list[dict] | None = None) -> tuple[str, list[dict]]:
        """Send one user turn, resolving any tool calls, return (reply_text, updated_conversation).

        Requires ANTHROPIC_API_KEY in the environment. This is the one part
        of the backend meant to be called from a real server process, not a
        browser artifact -- the accompanying demo artifact reimplements this
        same tool-calling loop client-side against mock data for a
        no-backend interactive demo (see project README).
        """
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set.")
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        messages = list(conversation) if conversation else []
        messages.append({"role": "user", "content": user_message})

        while True:
            resp = client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )
            messages.append({"role": "assistant", "content": resp.content})

            tool_uses = [b for b in resp.content if b.type == "tool_use"]
            if not tool_uses:
                text = "".join(b.text for b in resp.content if b.type == "text")
                return text, messages

            tool_results = []
            for tu in tool_uses:
                result = self._run_tool(tu.name, tu.input)
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": tu.id, "content": json.dumps(result)}
                )
            messages.append({"role": "user", "content": tool_results})
