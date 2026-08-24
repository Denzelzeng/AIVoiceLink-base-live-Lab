from __future__ import annotations

import struct
import unittest

from simultrans_baseline.audio import EnergyTurnSegmenter, pcm_to_wav
from simultrans_baseline.config import AudioConfig


def frame(amplitude: int, samples: int = 800) -> bytes:
    return struct.pack("<h", amplitude) * samples


class AudioTests(unittest.TestCase):
    def test_segmenter_emits_partial_and_final(self) -> None:
        config = AudioConfig(
            frame_ms=50,
            pre_roll_ms=50,
            min_speech_ms=100,
            end_silence_ms=100,
            partial_interval_ms=100,
            max_turn_ms=1000,
            energy_threshold=300,
            calibration_seconds=0,
        )
        segmenter = EnergyTurnSegmenter(config)
        self.assertEqual(segmenter.add_frame(frame(0)), [])
        self.assertEqual(segmenter.add_frame(frame(1200)), [])
        self.assertTrue(segmenter.consume_speech_started())
        outputs = []
        outputs.extend(segmenter.add_frame(frame(1200)))
        outputs.extend(segmenter.add_frame(frame(1200)))
        outputs.extend(segmenter.add_frame(frame(0)))
        outputs.extend(segmenter.add_frame(frame(0)))
        self.assertTrue(any(not value.is_final for value in outputs))
        self.assertTrue(outputs[-1].is_final)
        self.assertEqual(outputs[-1].sample_rate, 16_000)

    def test_pcm_to_wav_header(self) -> None:
        import io
        import wave

        value = pcm_to_wav(frame(1000))
        with wave.open(io.BytesIO(value), "rb") as stream:
            self.assertEqual(stream.getframerate(), 16_000)
            self.assertEqual(stream.getnchannels(), 1)
            self.assertEqual(stream.getsampwidth(), 2)


if __name__ == "__main__":
    unittest.main()

