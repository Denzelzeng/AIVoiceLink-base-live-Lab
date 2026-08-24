from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from simultrans_baseline.config import (
    ConfigurationError,
    TTSConfig,
    _read_workspace_csv,
    _validate_cloud_url,
    _workspace_service_base_url,
)


class ConfigTests(unittest.TestCase):
    def test_derives_workspace_realtime_tts_websocket_url(self) -> None:
        config = TTSConfig(
            "dashscope_qwen_voice_clone",
            "https://workspace.cn-beijing.maas.aliyuncs.com",
            "qwen3-tts-vc-realtime-2026-01-15",
            websocket_url="auto",
            speech_rate=1.3,
        )
        self.assertEqual(
            config.normalized_websocket_url,
            "wss://workspace.cn-beijing.maas.aliyuncs.com/api-ws/v1/realtime",
        )

    def test_reads_transposed_workspace_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workspace-apiKey.csv"
            path.write_text(
                "id,123\napiKey,secret\nworkspaceId,workspace\n"
                "openAiCompatible,https://example.test/v1\n",
                encoding="utf-8",
            )
            values = _read_workspace_csv(path)
        self.assertEqual(values["apiKey"], "secret")
        self.assertEqual(values["workspaceId"], "workspace")
        self.assertEqual(values["openAiCompatible"], "https://example.test/v1")

    def test_derives_native_api_origin_from_compatible_url(self) -> None:
        self.assertEqual(
            _workspace_service_base_url(
                "https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
            ),
            "https://workspace.cn-beijing.maas.aliyuncs.com",
        )

    def test_api_only_mode_rejects_local_or_insecure_model_urls(self) -> None:
        _validate_cloud_url("asr", "https://workspace.example.test/v1")
        for url in ("http://api.example.test/v1", "https://127.0.0.1:8004/v1"):
            with self.subTest(url=url), self.assertRaises(ConfigurationError):
                _validate_cloud_url("asr", url)


if __name__ == "__main__":
    unittest.main()
