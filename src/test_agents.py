from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config
from memory_store import UserProfileStore


def make_config(tmp_path: Path):
    """Student TODO: build an isolated config for tests."""

    # Hint:
    # - point `state_dir` into tmp_path
    # - reduce compact threshold so compaction happens quickly in tests
    config = load_config(tmp_path)
    config.compact_threshold_tokens = 80
    config.compact_keep_messages = 4
    return config


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Student TODO: verify `User.md` can be created, updated, and edited."""

    store = UserProfileStore(tmp_path / "profiles")
    path = store.write_text("dungct", "# User Profile\n\n- name: DũngCT\n")
    assert path.exists()
    assert "DũngCT" in store.read_text("dungct")
    assert store.edit_text("dungct", "DũngCT", "Dũng")
    assert "DũngCT" not in store.read_text("dungct")
    store.upsert_fact("dungct", "location", "Huế")
    assert store.facts("dungct")["location"] == "Huế"


def test_compact_trigger(tmp_path: Path) -> None:
    """Student TODO: verify long threads trigger compaction."""

    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    for index in range(12):
        agent.reply("user", "long", f"Lượt {index}: " + "ngữ cảnh dài " * 18)
    assert agent.compaction_count("long") > 0
    assert agent.compact_memory.context("long")["summary"]


def test_cross_session_recall(tmp_path: Path) -> None:
    """Student TODO: verify advanced remembers across sessions and baseline does not."""

    config = make_config(tmp_path)
    advanced = AdvancedAgent(config, force_offline=True)
    baseline = BaselineAgent(config, force_offline=True)
    fact = "Mình tên là DũngCT, hiện tại mình đang ở Huế."
    advanced.reply("dungct", "a-old", fact)
    baseline.reply("dungct", "b-old", fact)
    advanced_answer = advanced.reply("dungct", "a-new", "Mình tên gì và hiện tại ở đâu?")["answer"]
    baseline_answer = baseline.reply("dungct", "b-new", "Mình tên gì và hiện tại ở đâu?")["answer"]
    assert "DũngCT" in advanced_answer and "Huế" in advanced_answer
    assert "DũngCT" not in baseline_answer and "Huế" not in baseline_answer


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Student TODO: compare prompt load of baseline vs advanced on a long thread."""

    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    for index in range(20):
        message = f"Lượt {index}: " + "đây là nội dung dài để kiểm tra prompt memory " * 15
        baseline.reply("user", "baseline", message)
        advanced.reply("user", "advanced", message)
    assert advanced.compaction_count("advanced") > 0
    assert advanced.prompt_token_usage("advanced") < baseline.prompt_token_usage("baseline")
