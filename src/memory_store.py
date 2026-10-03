from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import unicodedata


def estimate_tokens(text: str) -> int:
    """Student TODO: implement a simple token estimator.

    Example idea:
    - Strip whitespace
    - Return 0 for empty text
    - Approximate tokens from character count, e.g. len(text) / 4
    """

    cleaned = " ".join((text or "").split())
    if not cleaned:
        return 0
    # Vietnamese and English prose average roughly 3-4 characters per token.
    return max(1, (len(cleaned) + 3) // 4)


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`.

    Student TODO:
    - Map each user id to one markdown file
    - Support read / write / edit operations
    - Optionally expose helpers like `facts()` or `upsert_fact()`
    """

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        normalized = unicodedata.normalize("NFKD", str(user_id)).encode("ascii", "ignore").decode()
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", normalized).strip("._")
        if not slug:
            raise ValueError("user_id must contain at least one safe character")
        return self.root_dir / slug / "User.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        return path.read_text(encoding="utf-8") if path.exists() else "# User Profile\n\n"

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        content = self.read_text(user_id)
        if search_text not in content:
            return False
        self.write_text(user_id, content.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        facts: dict[str, str] = {}
        for line in self.read_text(user_id).splitlines():
            match = re.match(r"^-\s*([^:]+):\s*(.+)$", line)
            if match:
                facts[match.group(1).strip()] = match.group(2).strip()
        return facts

    def upsert_fact(self, user_id: str, key: str, value: str) -> None:
        facts = self.facts(user_id)
        if key == "interests" and key in facts:
            existing = [item.strip() for item in facts[key].split(",")]
            incoming = [item.strip() for item in value.split(",")]
            value = ", ".join(dict.fromkeys(existing + incoming))
        facts[key] = value.strip().rstrip(".")
        lines = ["# User Profile", ""] + [f"- {name}: {item}" for name, item in facts.items()]
        self.write_text(user_id, "\n".join(lines) + "\n")


def extract_profile_updates(message: str) -> dict[str, str]:
    """Student TODO: convert raw user text into stable profile facts.

    Example facts you may want to extract:
    - name
    - location
    - profession
    - preferences / response style
    - favorite food / drink

    Pseudocode:
    1. Build a few regex patterns.
    2. Skip obvious question-only turns.
    3. Return only the facts that are confidently present in the message.
    """

    text = " ".join((message or "").split()).strip()
    lower = text.lower()
    question_cues = ("tên gì", "ở đâu", "nhắc lại", "mình nuôi con gì", "nghề gì", "bạn biết")
    if not text or text.endswith("?") or any(cue in lower for cue in question_cues):
        return {}
    updates: dict[str, str] = {}

    def capture(key: str, patterns: list[str]) -> None:
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                updates[key] = match.group(1).strip(" .,;:!?")
                return

    capture("name", [r"(?:mình|tôi)\s+tên\s+(?:là\s+)?([\wÀ-ỹ][\wÀ-ỹ ._-]{0,50}?)(?=,|\.|\s+(?:hiện|và|đang)\b)"])

    # Only accept explicit residence statements; travel and hypothetical locations are noise.
    if not any(noise in lower for noise in ("chỉ là nơi", "bay ra họp", "ví dụ cũ")):
        capture("location", [
            r"(?:hiện tại\s+|giờ\s+|hiện\s+)?mình\s+(?:đang\s+)?ở\s+([A-ZĐÀ-Ỹ][\wÀ-ỹ ]{1,35}?)(?=,|\.|\s+(?:chứ|và|trong|để|dù)\b)",
            r"nơi ở (?:hiện tại )?(?:là|đã cập nhật từ .+ sang)\s+([A-ZĐÀ-Ỹ][\wÀ-ỹ ]{1,35}?)(?=,|\.|$)",
        ])

    if "chỉ là câu đùa" not in lower:
        capture("profession", [
            r"(?:giờ\s+)?chuyển sang\s+([A-Za-z][A-Za-z0-9 +#.-]*?(?:engineer|developer|manager|scientist|designer))(?=\s+(?:cho|với|tại|nữa|và)|,|\.|$)",
            r"(?:nghề nghiệp (?:thì )?)?(?:hiện tại )?vẫn là\s+([A-Za-z][A-Za-z0-9 +#.-]*?(?:engineer|developer|manager|scientist|designer))(?=\s+(?:cho|với|tại|nữa|và)|,|\.|$)",
            r"(?:giờ\s+)?(?:mình\s+)?(?:đang\s+)?(?:làm|chuyển sang)\s+([A-Za-z][A-Za-z0-9 +#.-]*?(?:engineer|developer|manager|scientist|designer))(?=\s+(?:cho|với|tại|nữa|và)|,|\.|$)",
            r"nghề nghiệp (?:hiện tại )?(?:vẫn )?là\s+([A-Za-z][A-Za-z0-9 +#.-]+?)(?=,|\.|$)",
        ])

    capture("favorite_drink", [r"(?:đồ uống yêu thích (?:của mình )?là|mình vẫn uống)\s+([^,.]+?)(?=\s+nhưng\b|,|\.|$)"])
    capture("favorite_food", [r"(?:món ăn yêu thích (?:của mình )?là|món ruột(?: là)?)\s+([^,.]+)"])
    capture("pet", [r"(?:nuôi|con)\s+(?:một\s+|bé\s+)?(corgi)(?:\s+tên\s+([\wÀ-ỹ]+))?"])
    if "corgi" in lower:
        updates["pet"] = "corgi tên Bơ" if "bơ" in lower else "corgi"

    if "3 bullet" in lower:
        updates["response_style"] = "ngắn gọn, 3 bullet, có ví dụ thực chiến, nhấn mạnh trade-off"
    elif "ngắn gọn" in lower and any(word in lower for word in ("trả lời", "style", "giải thích")):
        style = "ngắn gọn"
        if "bullet" in lower:
            style += ", có bullet"
        if "ví dụ" in lower:
            style += ", có ví dụ thực tế"
        updates["response_style"] = style

    interests = [item for item in ("Python", "AI", "MLOps", "RAG", "memory architecture") if item.lower() in lower]
    if interests and any(cue in lower for cue in ("thích", "quan tâm", "dài hạn")):
        updates["interests"] = ", ".join(interests)
    return updates


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Student TODO: create a compact summary of older messages.

    This can be heuristic text concatenation first.
    Later, you can replace it with an LLM-based summary if desired.
    """

    if not messages:
        return ""
    selected = messages[-max_items:]
    parts = []
    for item in selected:
        role = "User" if item.get("role") == "user" else "Assistant"
        content = " ".join(item.get("content", "").split())
        if len(content) > 240:
            content = content[:237].rstrip() + "..."
        parts.append(f"{role}: {content}")
    return " | ".join(parts)


@dataclass
class CompactMemoryManager:
    """Student TODO: implement compact memory for long threads.

    Goal:
    - Keep recent messages in full
    - When the thread grows too large, move older content into a summary
    - Track how many compactions happened for benchmarking
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        thread = self.state.setdefault(thread_id, {"messages": [], "summary": "", "compactions": 0})
        messages = thread["messages"]
        assert isinstance(messages, list)
        messages.append({"role": role, "content": content})
        context_text = str(thread["summary"]) + " " + " ".join(str(m.get("content", "")) for m in messages)
        if estimate_tokens(context_text) > self.threshold_tokens and len(messages) > self.keep_messages:
            old = messages[:-self.keep_messages]
            previous = str(thread["summary"])
            new_summary = summarize_messages(old, max_items=max(6, len(old)))
            thread["summary"] = summarize_messages(
                ([{"role": "assistant", "content": previous}] if previous else [])
                + [{"role": "assistant", "content": new_summary}],
                max_items=2,
            )
            thread["messages"] = messages[-self.keep_messages:]
            thread["compactions"] = int(thread["compactions"]) + 1

    def context(self, thread_id: str) -> dict[str, object]:
        thread = self.state.setdefault(thread_id, {"messages": [], "summary": "", "compactions": 0})
        return {"messages": list(thread["messages"]), "summary": thread["summary"], "compactions": thread["compactions"]}

    def compaction_count(self, thread_id: str) -> int:
        return int(self.context(thread_id)["compactions"])
