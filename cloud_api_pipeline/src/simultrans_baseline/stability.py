from __future__ import annotations

import re
from collections import deque
from collections.abc import Sequence

from .events import CommitUpdate


_UNIT_PATTERN = re.compile(r"\s+|[\u3400-\u9fff]|[\w]+|[^\w\s]", re.UNICODE)
_BOUNDARY_PATTERN = re.compile(r"[.!?;:。！？；：]\s*$")


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def text_units(text: str) -> list[str]:
    return _UNIT_PATTERN.findall(text)


def longest_common_prefix(values: Sequence[str]) -> str:
    if not values:
        return ""
    units = [text_units(value) for value in values]
    limit = min(len(value) for value in units)
    index = 0
    while index < limit and all(value[index] == units[0][index] for value in units[1:]):
        index += 1
    return "".join(units[0][:index]).rstrip()


def _append_monotonic(committed: str, hypothesis: str) -> str:
    if not committed:
        return hypothesis
    if hypothesis.startswith(committed):
        return hypothesis
    overlap = longest_common_prefix([committed, hypothesis])
    suffix = hypothesis[len(overlap) :].lstrip()
    if not suffix:
        return committed
    separator = "" if committed[-1:].isspace() or suffix[:1] in ",.!?;:，。！？；：" else " "
    return committed + separator + suffix


class AgreementCommitter:
    """Strict LocalAgreement/LCP committer with an irreversible prefix."""

    def __init__(self, depth: int = 2):
        if depth < 1:
            raise ValueError("agreement depth must be positive")
        self.depth = depth
        self.committed = ""
        self._history: deque[str] = deque(maxlen=depth)

    def update(self, hypothesis: str, *, is_final: bool = False) -> CommitUpdate:
        hypothesis = normalize_text(hypothesis)
        old = self.committed
        if is_final:
            self.committed = _append_monotonic(self.committed, hypothesis)
            self._history.clear()
        else:
            self._history.append(hypothesis)
            if len(self._history) >= self.depth:
                stable = longest_common_prefix(list(self._history))
                if stable.startswith(self.committed) and len(stable) > len(self.committed):
                    self.committed = stable

        if hypothesis.startswith(self.committed):
            unstable = hypothesis[len(self.committed) :].lstrip()
        elif is_final:
            unstable = ""
        else:
            unstable = hypothesis
        return CommitUpdate(
            committed=self.committed,
            unstable=unstable,
            commit_delta=self.committed[len(old) :],
            is_final=is_final,
            hypothesis=hypothesis,
        )


class PhraseBuffer:
    """Only releases committed target text at safe TTS phrase boundaries."""

    def __init__(self, min_chars: int = 8):
        if min_chars < 1:
            raise ValueError("min_chars must be positive")
        self.min_chars = min_chars
        self._buffer = ""

    def push(self, delta: str, *, is_final: bool = False) -> list[str]:
        self._buffer += delta
        value = self._buffer.strip()
        if not value:
            return []
        if is_final or (len(value) >= self.min_chars and _BOUNDARY_PATTERN.search(value)):
            self._buffer = ""
            return [value]
        return []

    def flush(self) -> list[str]:
        return self.push("", is_final=True)

