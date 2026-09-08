from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from nexus_learning.closure_effectiveness import (
    append_learning_episode,
    load_learning_closures,
    normalize_learning_episode,
)


def _episode(episode_id: str) -> dict[str, object]:
    episode = normalize_learning_episode(task_id=episode_id, attempt_id=f"attempt-{episode_id}")
    episode["episode_id"] = episode_id
    return episode


@pytest.mark.parametrize("initial", [b"", b"\n", b"  \n\n"])
def test_append_normal_and_empty_history_is_reader_visible(tmp_path: Path, initial: bytes) -> None:
    path = tmp_path / "learning_episodes.jsonl"
    path.write_bytes(initial)

    assert append_learning_episode(path, _episode("A")) is True
    assert [row["episode_id"] for row in load_learning_closures(path)] == ["A"]


def test_append_frames_unterminated_complete_tail_and_preserves_all_rows(tmp_path: Path) -> None:
    path = tmp_path / "learning_episodes.jsonl"
    first = _episode("A")
    second = _episode("B")
    third = _episode("C")

    assert append_learning_episode(path, first) is True
    path.write_bytes(path.read_bytes() + json.dumps(second).encode("utf-8"))
    assert append_learning_episode(path, third) is True

    assert [row["episode_id"] for row in load_learning_closures(path)] == ["A", "B", "C"]


@pytest.mark.parametrize("tail", [b'{"episode_id":"partial"', b"not-json", b"[]"])
def test_append_rejects_unterminated_invalid_tail_without_mutation(
    tmp_path: Path, tail: bytes
) -> None:
    path = tmp_path / "learning_episodes.jsonl"
    path.write_bytes(tail)
    before = path.read_bytes()

    assert append_learning_episode(path, _episode("C")) is False
    assert path.read_bytes() == before


def test_duplicate_is_checked_after_tail_validation(tmp_path: Path) -> None:
    path = tmp_path / "learning_episodes.jsonl"
    episode = _episode("A")
    assert append_learning_episode(path, episode) is True
    path.write_bytes(path.read_bytes() + b"broken")
    before = path.read_bytes()

    assert append_learning_episode(path, episode) is False
    assert path.read_bytes() == before


def test_duplicate_retry_does_not_append_second_record(tmp_path: Path) -> None:
    path = tmp_path / "learning_episodes.jsonl"
    episode = _episode("A")

    assert append_learning_episode(path, episode) is True
    before = path.read_bytes()
    assert append_learning_episode(path, episode) is True
    assert path.read_bytes() == before
    assert [row["episode_id"] for row in load_learning_closures(path)] == ["A"]


def test_concurrent_unique_appends_remain_reader_visible(tmp_path: Path) -> None:
    path = tmp_path / "learning_episodes.jsonl"
    episodes = [_episode(f"episode-{index}") for index in range(20)]

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda episode: append_learning_episode(path, episode), episodes))

    assert results == [True] * len(episodes)
    rows = load_learning_closures(path)
    assert {row["episode_id"] for row in rows} == {episode["episode_id"] for episode in episodes}
