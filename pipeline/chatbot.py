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

Backend: this calls a Hugging Face-hosted instruct model through the free
HF Inference API instead of a paid LLM API. `HF_CHAT_MODEL` defaults to
`meta-llama/Llama-3.1-8B-Instruct`; pick any other tool-calling-capable
instruct model on the Hub (or a self-hosted TGI/vLLM endpoint via
`HF_ENDPOINT_URL`) without changing the call site. Tool calling here follows
the OpenAI function-calling schema (that's what HF's Inference API speaks),
so `TOOLS` below is defined once in Anthropic's simpler `input_schema` shape
and reshaped by `_tools_to_openai_schema()` -- avoids maintaining two nearly
identical copies of the same three tool definitions.
"""
from __future__ import annotations

import json
import os

from .storage import Storage
from .latency import LATENCY

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

HF_CHAT_MODEL = os.environ.get("HF_CHAT_MODEL", "meta-llama/Llama-3.1-8B-Instruct")


def _tools_to_openai_schema(tools: list[dict]) -> list[dict]:
    """Reshape the Anthropic-style TOOLS list above into the OpenAI-style
    function-calling schema HF's chat_completion API expects."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in tools
    ]


def _serialize_tool_calls(tool_calls) -> list[dict] | None:
    """Turn the SDK's tool_call objects back into plain dicts so they can be
    round-tripped in the `messages` list on the next loop iteration."""
    if not tool_calls:
        return None
    return [
        {
            "id": tc.id,
            "type": "function",
            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
        }
        for tc in tool_calls
    ]


class NutritionChatbot:
    def __init__(self, storage: Storage, user_id: str, model: str = HF_CHAT_MODEL):
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

        Requires HF_TOKEN in the environment -- a free Hugging Face account
        token (https://huggingface.co/settings/tokens). This is the one part
        of the backend meant to be called from a real server process, not a
        browser artifact -- the accompanying demo artifact reimplements this
        same tool-calling loop client-side against mock data for a
        no-backend interactive demo (see project README).

        Each full turn (which may include one or more tool round-trips) is
        timed as a single "chatbot_turn" sample in the shared `LATENCY`
        tracker. For a chat UI, p95/p99 on a full turn -- not just the raw
        model-call latency -- is the number that maps to what a user
        actually waits for, including tool-call round-trips.
        """
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise RuntimeError("HF_TOKEN not set.")
        from huggingface_hub import InferenceClient

        endpoint = os.environ.get("HF_ENDPOINT_URL")  # optional: self-hosted TGI/vLLM endpoint
        client = InferenceClient(model=endpoint or self.model, token=hf_token)

        messages = list(conversation) if conversation else [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.append({"role": "user", "content": user_message})
        openai_tools = _tools_to_openai_schema(TOOLS)

        with LATENCY.track("chatbot_turn"):
            while True:
                completion = client.chat_completion(
                    messages=messages,
                    tools=openai_tools,
                    tool_choice="auto",
                    max_tokens=1024,
                )
                msg = completion.choices[0].message

                assistant_msg = {"role": "assistant", "content": msg.content or ""}
                serialized_calls = _serialize_tool_calls(msg.tool_calls)
                if serialized_calls:
                    assistant_msg["tool_calls"] = serialized_calls
                messages.append(assistant_msg)

                if not msg.tool_calls:
                    return msg.content or "", messages

                for tc in msg.tool_calls:
                    args = tc.function.arguments
                    if isinstance(args, str):
                        args = json.loads(args)
                    result = self._run_tool(tc.function.name, args)
                    messages.append(
                        {"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)}
                    )
