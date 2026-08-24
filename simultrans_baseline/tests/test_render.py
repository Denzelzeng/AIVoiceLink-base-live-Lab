from __future__ import annotations

import io
import unittest

from simultrans_baseline.events import PipelineEvent
from simultrans_baseline.render import ConsoleRenderer


class _TTYBuffer(io.StringIO):
    def isatty(self) -> bool:
        return True


def event(kind: str, *, turn_id: int = 1, **data) -> PipelineEvent:
    return PipelineEvent(
        kind=kind,
        session_id="test-session",
        turn_id=turn_id,
        data=data,
    )


class ConsoleRendererTests(unittest.TestCase):
    def test_non_tty_prints_only_one_final_line_per_turn(self) -> None:
        stream = io.StringIO()
        renderer = ConsoleRenderer(stream)
        renderer(event("transcript.update", committed="", unstable="试", is_final=False))
        renderer(event("transcript.update", committed="试", unstable="一下", is_final=False))
        renderer(event("translation.update", committed="", speculative="Try", is_final=False))
        renderer(event("endpoint.decision", complete=True, reason="punctuation"))
        renderer(event("transcript.update", committed="试一下。", unstable="", is_final=True))
        renderer(event("translation.update", committed="Try it.", speculative="", is_final=True))
        renderer(event("tts.started", text="Try it."))

        output = stream.getvalue()
        self.assertEqual(output.count("[同传 #1]"), 1)
        self.assertEqual(output.count("Try it."), 1)
        self.assertNotIn("partial", output)
        self.assertNotIn("语义端点", output)
        self.assertIn("[语音 #1] 开始播放", output)

    def test_tty_partials_rewrite_one_live_line(self) -> None:
        stream = _TTYBuffer()
        renderer = ConsoleRenderer(stream)
        renderer(event("transcript.update", committed="", unstable="试", is_final=False))
        renderer(event("transcript.update", committed="试", unstable="一下", is_final=False))
        renderer(event("translation.update", committed="", speculative="Try it", is_final=False))
        renderer(event("transcript.update", committed="试一下。", unstable="", is_final=True))
        renderer(event("translation.update", committed="Try it.", speculative="", is_final=True))

        output = stream.getvalue()
        self.assertIn("\r\x1b[2K", output)
        self.assertEqual(output.count("\n"), 1)
        self.assertIn("[同传 #1] 试一下。  →  Try it.", output)


if __name__ == "__main__":
    unittest.main()
