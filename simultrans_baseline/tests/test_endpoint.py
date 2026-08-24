from __future__ import annotations

import unittest

import httpx

from simultrans_baseline.config import EndpointConfig
from simultrans_baseline.endpoint import (
    FallbackSemanticEndpoint,
    HeuristicSemanticEndpoint,
    LLMSemanticEndpoint,
)
from simultrans_baseline.events import AudioWindow


class EndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_continuation_and_complete_cues(self) -> None:
        endpoint = HeuristicSemanticEndpoint()
        window = AudioWindow(1, b"\0\0" * 1600, 16000, True, 0.0, 1.0)
        incomplete = await endpoint.classify(
            window, transcript="因为", language="Chinese"
        )
        complete = await endpoint.classify(
            window, transcript="天气很好。", language="Chinese"
        )
        self.assertFalse(incomplete.complete)
        self.assertTrue(complete.complete)

    async def test_remote_llm_classifies_incomplete_pause(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "INCOMPLETE"}}]},
            )

        endpoint = LLMSemanticEndpoint(
            EndpointConfig(
                provider="llm_http",
                base_url="https://api.test/v1",
                model="qwen-flash",
            ),
            transport=httpx.MockTransport(handler),
        )
        window = AudioWindow(1, b"\0\0" * 1600, 16000, True, 0.0, 1.0)
        decision = await endpoint.classify(
            window, transcript="因为", language="Chinese"
        )
        self.assertFalse(decision.complete)
        await endpoint.aclose()

    async def test_semantic_api_failure_falls_back_without_stopping_pipeline(self) -> None:
        class Broken:
            async def classify(self, *args, **kwargs):
                raise RuntimeError("offline")

            async def health(self):
                raise RuntimeError("offline")

            async def aclose(self):
                return None

        endpoint = FallbackSemanticEndpoint(Broken())
        window = AudioWindow(1, b"\0\0" * 1600, 16000, True, 0.0, 1.0)
        decision = await endpoint.classify(
            window, transcript="天气很好。", language="Chinese"
        )
        self.assertTrue(decision.complete)
        self.assertIn("fallback", decision.reason)


if __name__ == "__main__":
    unittest.main()
