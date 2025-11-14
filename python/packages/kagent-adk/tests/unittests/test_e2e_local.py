import json
import os
import sys
from pathlib import Path
from typing import AsyncGenerator

import httpx
import pytest
from a2a.types import AgentCard
from fastapi import FastAPI
from google.adk.agents import BaseAgent
from google.adk.events import Event
from google.genai import types

# Add src/ to path
_PKG_ROOT = Path(__file__).resolve().parents[2]  # .../packages/kagent-adk
_SRC = _PKG_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from kagent.adk import KAgentApp  # noqa: E402


class EchoAgent(BaseAgent):
    """Simple agent that echoes the user message with a prefix."""

    async def run_async(self, message: types.Content) -> AsyncGenerator[Event, None]:
        user_text = "".join(part.text for part in message.parts if hasattr(part, "text"))
        response = f"[ECHO] {user_text}"
        yield Event(
            author="agent",
            content=types.Content(role="assistant", parts=[types.Part(text=response)]),
            is_final_response=True,
        )


@pytest.fixture
def agent_card() -> AgentCard:
    return AgentCard(
        name="echo-agent",
        description="An agent that echoes your message",
        version="0.1.0",
        author="test",
        homepage="",
        tags=[],
    )


@pytest.fixture
def kagent_app_local(agent_card: AgentCard) -> KAgentApp:
    return KAgentApp(
        root_agent=EchoAgent(),
        agent_card=agent_card,
        kagent_url="http://localhost:8083",
        app_name="test-echo",
    )


@pytest.fixture
async def fastapi_app(kagent_app_local: KAgentApp) -> FastAPI:
    return kagent_app_local.build_local()


@pytest.fixture
async def client(fastapi_app: FastAPI) -> AsyncGenerator[httpx.AsyncClient, None]:
    async with httpx.AsyncClient(app=fastapi_app, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_build_local_health_check(client: httpx.AsyncClient):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.text == "OK"


@pytest.mark.asyncio
async def test_build_local_thread_dump(client: httpx.AsyncClient):
    response = await client.get("/thread_dump")
    assert response.status_code == 200
    assert "Thread" in response.text or "Traceback" in response.text


@pytest.mark.asyncio
async def test_build_local_a2a_send_message(client: httpx.AsyncClient):
    payload = {
        "message": {
            "kind": "message",
            "role": "user",
            "parts": [{"text": "Hello, world!"}]
        }
    }

    response = await client.post("/api/a2a/v1/send_message", json=payload)
    assert response.status_code == 200

    result = response.json()
    assert "result" in result
    task = result["result"].get("task", {})
    assert "history" in task

    # Find the last assistant message
    assistant_texts = [
        part.get("text", "")
        for msg in task["history"]
        if msg.get("role") == "assistant"
        for part in msg.get("parts", [])
        if part.get("type") == "text"
    ]

    assert len(assistant_texts) > 0
    assert "[ECHO] Hello, world!" in assistant_texts[-1]


@pytest.mark.asyncio
async def test_build_local_a2a_stream_message(client: httpx.AsyncClient):
    payload = {
        "message": {
            "kind": "message",
            "role": "user",
            "parts": [{"text": "Stream me!"}]
        }
    }

    async with client.stream("POST", "/api/a2a/v1/stream_message", json=payload) as response:
        assert response.status_code == 200

        full_text = ""
        async for line in response.aiter_lines():
            if not line.strip():
                continue
            # SSE format: data: {...}
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    break
                event = json.loads(data)
                if "result" in event:
                    status = event["result"].get("task_status_update", {})
                    message = status.get("message")
                    if message and message.get("role") == "assistant":
                        for part in message.get("parts", []):
                            if part.get("type") == "text":
                                full_text += part["text"]

        assert "[ECHO] Stream me!" in full_text