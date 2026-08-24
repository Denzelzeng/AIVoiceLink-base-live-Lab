from __future__ import annotations

import unittest
from collections import deque
from dataclasses import replace

from simultrans_baseline.config import SpeakerChangeConfig
from simultrans_baseline.speaker import SpeakerChangeDetector, cosine_similarity


class FakeEmbedder:
    def __init__(self, embeddings):
        self.embeddings = deque(embeddings)

    def embed(self, pcm, *, sample_rate):
        del pcm, sample_rate
        return self.embeddings.popleft()


class SpeakerChangeTests(unittest.IsolatedAsyncioTestCase):
    async def test_same_speaker_does_not_trigger_change(self) -> None:
        detector = SpeakerChangeDetector(
            FakeEmbedder([[1.0, 0.0], [0.99, 0.01]]),
            SpeakerChangeConfig(enabled=True, model_path="mock"),
        )
        await detector.seed(b"seed", sample_rate=16_000)
        result = await detector.assess(b"same", sample_rate=16_000)
        self.assertEqual(result.state, "same")
        self.assertFalse(result.changed)
        self.assertGreater(result.similarity or 0, 0.99)

    async def test_different_speaker_triggers_change(self) -> None:
        detector = SpeakerChangeDetector(
            FakeEmbedder([[1.0, 0.0], [0.0, 1.0], [0.01, 0.99]]),
            SpeakerChangeConfig(enabled=True, model_path="mock"),
        )
        await detector.seed(b"seed", sample_rate=16_000)
        changed = await detector.assess(b"new", sample_rate=16_000)
        following = await detector.assess(b"new-again", sample_rate=16_000)
        self.assertTrue(changed.changed)
        self.assertEqual(following.state, "same")

    async def test_ambiguous_score_keeps_current_speaker(self) -> None:
        detector = SpeakerChangeDetector(
            FakeEmbedder([[1.0, 0.0], [0.7, 0.7]]),
            SpeakerChangeConfig(enabled=True, model_path="mock"),
        )
        await detector.seed(b"seed", sample_rate=16_000)
        result = await detector.assess(b"ambiguous", sample_rate=16_000)
        self.assertEqual(result.state, "ambiguous")
        self.assertFalse(result.changed)

    async def test_optional_multi_window_confirmation(self) -> None:
        config = replace(
            SpeakerChangeConfig(enabled=True, model_path="mock"),
            confirmation_windows=2,
        )
        detector = SpeakerChangeDetector(
            FakeEmbedder([[1.0, 0.0], [0.0, 1.0], [0.01, 0.99]]),
            config,
        )
        await detector.seed(b"seed", sample_rate=16_000)
        first = await detector.assess(b"new-1", sample_rate=16_000)
        second = await detector.assess(b"new-2", sample_rate=16_000)
        self.assertEqual(first.state, "candidate")
        self.assertFalse(first.changed)
        self.assertTrue(second.changed)

    def test_cosine_similarity_normalizes_embeddings(self) -> None:
        self.assertAlmostEqual(cosine_similarity([2, 0], [3, 0]), 1.0)
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0.0)


if __name__ == "__main__":
    unittest.main()
