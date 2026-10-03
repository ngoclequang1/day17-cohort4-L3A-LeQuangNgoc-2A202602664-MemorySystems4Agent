from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Student TODO: implement Agent A.

    Requirements:
    - Within-session memory only
    - No persistent `User.md`
    - Should forget long-term facts across new threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}

        # TODO: optionally initialize a real LangChain/LangGraph agent when dependencies exist.
        self.langchain_agent = None if force_offline else self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Student TODO: return the agent response and token accounting.

        Pseudocode:
        - If a live agent exists, call the live path.
        - Otherwise use a deterministic offline path.
        """

        if self.langchain_agent is None:
            return self._reply_offline(thread_id, message)
        session = self.sessions.setdefault(thread_id, SessionState())
        session.messages.append({"role": "user", "content": message})
        session.prompt_tokens_processed += sum(estimate_tokens(m["content"]) for m in session.messages)
        response = self.langchain_agent.invoke(session.messages)
        answer = response.content if hasattr(response, "content") else str(response)
        session.messages.append({"role": "assistant", "content": answer})
        session.token_usage += estimate_tokens(answer)
        return {"answer": answer, "tokens": estimate_tokens(answer), "prompt_tokens": session.prompt_tokens_processed}

    def token_usage(self, thread_id: str) -> int:
        # TODO: return cumulative agent token count for one thread.
        return self.sessions.get(thread_id, SessionState()).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        # TODO: estimate how much prompt context this baseline kept processing.
        return self.sessions.get(thread_id, SessionState()).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Student TODO: implement a simple offline behavior.

        Suggested behavior:
        - Store the new user message in the session
        - Generate a short deterministic reply
        - Update token counts
        - Never remember facts across different thread ids
        """

        session = self.sessions.setdefault(thread_id, SessionState())
        session.messages.append({"role": "user", "content": message})
        prompt_tokens = sum(estimate_tokens(item["content"]) for item in session.messages)
        session.prompt_tokens_processed += prompt_tokens

        lower = message.lower()
        if any(term in lower for term in ("tên gì", "nhắc lại", "hiện tại", "yêu thích", "style", "nghề")):
            answer = "Mình chỉ nhớ thông tin trong phiên hiện tại và chưa có đủ dữ liệu để trả lời chắc chắn."
        else:
            answer = "Mình đã ghi nhận thông tin này trong phiên hiện tại."
        session.messages.append({"role": "assistant", "content": answer})
        answer_tokens = estimate_tokens(answer)
        session.token_usage += answer_tokens
        return {"answer": answer, "tokens": answer_tokens, "prompt_tokens": prompt_tokens}

    def _maybe_build_langchain_agent(self):
        """Student TODO: optionally wire `create_agent` + `InMemorySaver` here.

        Use `build_chat_model(self.config.model)` so the baseline can run with any supported provider.
        """

        # A key is the explicit opt-in for remote execution; otherwise offline mode is safe.
        if not self.config.model.api_key and self.config.model.provider != "ollama":
            return None
        try:
            return build_chat_model(self.config.model)
        except (ImportError, RuntimeError, ValueError):
            return None
