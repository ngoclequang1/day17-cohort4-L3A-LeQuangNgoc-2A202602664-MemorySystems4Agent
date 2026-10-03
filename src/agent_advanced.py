from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Student TODO: implement Agent B / Advanced Agent.

    Required memory layers:
    1. within-session memory
    2. persistent `User.md`
    3. compact memory for long threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}

        # TODO: optionally initialize a real LangChain/LangGraph agent.
        self.langchain_agent = None if force_offline else self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Student TODO: route between offline mode and live mode."""

        if self.langchain_agent is None:
            return self._reply_offline(user_id, thread_id, message)

        updates = extract_profile_updates(message)
        for key, value in updates.items():
            self.profile_store.upsert_fact(user_id, key, value)
        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        context = self.compact_memory.context(thread_id)
        system = "Use this persistent user profile and compact summary.\n" + self.profile_store.read_text(user_id)
        if context["summary"]:
            system += "\nSummary: " + str(context["summary"])
        live_messages = [{"role": "system", "content": system}] + list(context["messages"])
        response = self.langchain_agent.invoke(live_messages)
        answer = response.content if hasattr(response, "content") else str(response)
        self.compact_memory.append(thread_id, "assistant", answer)
        used = estimate_tokens(answer)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + used
        return {"answer": answer, "tokens": used, "prompt_tokens": prompt_tokens}

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Student TODO: implement the deterministic advanced path.

        Pseudocode:
        1. Extract stable profile facts from the incoming message.
        2. Persist those facts into `User.md`.
        3. Append the message into compact memory.
        4. Estimate prompt-context load from `User.md` + summary + recent messages.
        5. Generate a response that can answer long-term recall questions.
        6. Append the assistant reply and update token counters.
        """

        updates = extract_profile_updates(message)
        for key, value in updates.items():
            self.profile_store.upsert_fact(user_id, key, value)

        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        answer = self._offline_response(user_id, thread_id, message)
        self.compact_memory.append(thread_id, "assistant", answer)
        answer_tokens = estimate_tokens(answer)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + answer_tokens
        return {"answer": answer, "tokens": answer_tokens, "prompt_tokens": prompt_tokens}

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Student TODO: estimate the context carried into one turn.

        Hint:
        - Include `User.md`
        - Include compact summary text
        - Include recent kept messages
        """

        context = self.compact_memory.context(thread_id)
        recent = " ".join(str(item.get("content", "")) for item in context["messages"])
        return estimate_tokens(self.profile_store.read_text(user_id)) + estimate_tokens(str(context["summary"])) + estimate_tokens(recent)

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        """Student TODO: return a deterministic answer using persisted memory.

        Make sure the advanced agent can answer questions like:
        - "Mình tên gì?"
        - "Hiện tại mình làm nghề gì?"
        - "Nhắc lại style trả lời mình thích"
        - questions in the long stress dataset
        """

        facts = self.profile_store.facts(user_id)
        lower = message.lower()
        requested: list[str] = []
        mapping = {
            "name": ("tên",),
            "profession": ("nghề", "làm nghề", "là ai"),
            "location": ("ở đâu", "nơi ở", "huế", "hà nội"),
            "response_style": ("style", "kiểu trả lời"),
            "favorite_drink": ("đồ uống",),
            "favorite_food": ("món ăn",),
            "pet": ("nuôi con", "corgi"),
            "interests": ("mối quan tâm", "kỹ thuật chính", "tóm tắt"),
        }
        for key, cues in mapping.items():
            if key in facts and any(cue in lower for cue in cues):
                requested.append(facts[key])
        if not requested and any(cue in lower for cue in ("nhắc lại", "tóm tắt", "biết")):
            requested = list(facts.values())
        if requested:
            return "Mình nhớ: " + "; ".join(dict.fromkeys(requested)) + "."
        return "Mình đã cập nhật thông tin ổn định vào hồ sơ và sẽ ưu tiên bản mới nhất."

    def _maybe_build_langchain_agent(self):
        """Student TODO: wire a live agent with tools and compact middleware.

        High-level design:
        - `build_chat_model(self.config.model)` for the selected provider
        - `InMemorySaver` for short-term thread state
        - tool to read `User.md`
        - tool to write/edit `User.md`
        - dynamic prompt that injects profile memory
        - summarization middleware for long threads
        """

        if not self.config.model.api_key and self.config.model.provider != "ollama":
            return None
        try:
            return build_chat_model(self.config.model)
        except (ImportError, RuntimeError, ValueError):
            return None
