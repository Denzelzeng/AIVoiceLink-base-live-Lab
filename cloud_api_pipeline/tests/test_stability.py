from __future__ import annotations

import unittest

from simultrans_baseline.stability import (
    AgreementCommitter,
    PhraseBuffer,
    longest_common_prefix,
)


class StabilityTests(unittest.TestCase):
    def test_chinese_and_english_common_prefix(self) -> None:
        self.assertEqual(longest_common_prefix(["大家好，欢迎", "大家好，晚上好"]), "大家好，")
        self.assertEqual(longest_common_prefix(["Hello world", "Hello there"]), "Hello")

    def test_strict_agreement_never_rolls_back_committed_text(self) -> None:
        committer = AgreementCommitter(depth=2)
        first = committer.update("Hello everyone")
        self.assertEqual(first.committed, "")
        second = committer.update("Hello everyone, welcome")
        self.assertEqual(second.committed, "Hello everyone")
        conflict = committer.update("Good evening, welcome")
        self.assertEqual(conflict.committed, "Hello everyone")
        final = committer.update("Good evening, welcome.", is_final=True)
        self.assertTrue(final.committed.startswith("Hello everyone"))

    def test_phrase_buffer_only_releases_safe_boundary_or_final(self) -> None:
        buffer = PhraseBuffer(min_chars=5)
        self.assertEqual(buffer.push("Hello"), [])
        self.assertEqual(buffer.push(" world."), ["Hello world."])
        self.assertEqual(buffer.push("Tail", is_final=True), ["Tail"])


if __name__ == "__main__":
    unittest.main()

