from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
import json
import tempfile

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Student TODO: read JSON conversations from disk."""

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of conversations in {path}")
    return data


def recall_points(answer: str, expected: list[str]) -> float:
    """Student TODO: return 0 / 0.5 / 1 depending on how many expected facts appear."""

    if not expected:
        return 1.0
    normalized = answer.casefold()
    found = sum(1 for item in expected if item.casefold() in normalized)
    ratio = found / len(expected)
    return 1.0 if ratio == 1 else 0.5 if ratio >= 0.5 else 0.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Student TODO: add a lightweight quality score for offline mode."""

    recall = recall_points(answer, expected)
    if not answer.strip():
        return 0.0
    clarity = 1.0 if 5 <= len(answer.split()) <= 100 else 0.5
    return round(0.8 * recall + 0.2 * clarity, 2)


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    """Student TODO: evaluate one agent over many conversations.

    Pseudocode:
    1. Feed all turns to the agent.
    2. Track `agent tokens only`.
    3. Track `prompt tokens processed`.
    4. Ask recall questions in a fresh thread.
    5. Compute average recall and quality.
    6. Record memory file growth and compaction count.
    """

    recall_scores: list[float] = []
    quality_scores: list[float] = []
    thread_ids: list[str] = []
    users: set[str] = set()
    before_sizes: dict[str, int] = {}

    for conversation in conversations:
        user_id = str(conversation["user_id"])
        users.add(user_id)
        if hasattr(agent, "memory_file_size") and user_id not in before_sizes:
            before_sizes[user_id] = agent.memory_file_size(user_id)
        thread_id = f"{agent_name}-{conversation['id']}"
        thread_ids.append(thread_id)
        for turn in conversation.get("turns", []):
            agent.reply(user_id, thread_id, str(turn))
        for index, item in enumerate(conversation.get("recall_questions", [])):
            recall_thread = f"{thread_id}-recall-{index}"
            thread_ids.append(recall_thread)
            result = agent.reply(user_id, recall_thread, item["question"])
            answer = result["answer"]
            expected = list(item.get("expected_contains", []))
            recall_scores.append(recall_points(answer, expected))
            quality_scores.append(heuristic_quality(answer, expected))

    growth = 0
    if hasattr(agent, "memory_file_size"):
        growth = sum(max(0, agent.memory_file_size(uid) - before_sizes.get(uid, 0)) for uid in users)
    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=sum(agent.token_usage(tid) for tid in thread_ids),
        prompt_tokens_processed=sum(agent.prompt_token_usage(tid) for tid in thread_ids),
        recall_score=round(sum(recall_scores) / len(recall_scores), 3) if recall_scores else 0.0,
        response_quality=round(sum(quality_scores) / len(quality_scores), 3) if quality_scores else 0.0,
        memory_growth_bytes=growth,
        compactions=sum(agent.compaction_count(tid) for tid in thread_ids),
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Student TODO: print a markdown table or tabulated output."""

    headers = ["Agent", "Agent tokens only", "Prompt tokens processed", "Cross-session recall", "Response quality", "Memory growth (bytes)", "Compactions"]
    values = [[r.agent_name, r.agent_tokens_only, r.prompt_tokens_processed, f"{r.recall_score:.2f}", f"{r.response_quality:.2f}", r.memory_growth_bytes, r.compactions] for r in rows]
    try:
        from tabulate import tabulate

        return tabulate(values, headers=headers, tablefmt="github")
    except ImportError:
        all_rows = [headers] + [[str(value) for value in row] for row in values]
        widths = [max(len(str(row[i])) for row in all_rows) for i in range(len(headers))]
        render = lambda row: "| " + " | ".join(str(value).ljust(widths[i]) for i, value in enumerate(row)) + " |"
        return "\n".join([render(headers), "| " + " | ".join("-" * width for width in widths) + " |"] + [render(row) for row in values])


def main() -> None:
    """Student TODO: run both benchmark suites.

    Required benchmark sections:
    - Standard benchmark from `data/conversations.json`
    - Long-context stress benchmark from `data/advanced_long_context.json`

    Compare:
    - Baseline
    - Advanced

    Keep the same output columns as the solved lab:
    - Agent tokens only
    - Prompt tokens processed
    - Cross-session recall
    - Response quality
    - Memory growth (bytes)
    - Compactions
    """

    config = load_config(Path(__file__).resolve().parent.parent)

    suites = [
        ("Standard Benchmark", config.data_dir / "conversations.json"),
        ("Long-Context Stress Benchmark", config.data_dir / "advanced_long_context.json"),
    ]
    for title, path in suites:
        conversations = load_conversations(path)
        # Each suite gets fresh persistent storage so repeated benchmark runs are reproducible.
        with tempfile.TemporaryDirectory(prefix="benchmark-", dir=config.state_dir) as temp_dir:
            suite_config = replace(config, state_dir=Path(temp_dir))
            baseline = BaselineAgent(suite_config, force_offline=True)
            advanced = AdvancedAgent(suite_config, force_offline=True)
            rows = [
                run_agent_benchmark("Baseline", baseline, conversations, suite_config),
                run_agent_benchmark("Advanced", advanced, conversations, suite_config),
            ]
        print(f"\n## {title}\n")
        print(format_rows(rows))


if __name__ == "__main__":
    main()
