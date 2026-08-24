from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    from .qwen3_omni_client import (
        ASRConfig,
        ClientConfig,
        Qwen3ASRClient,
        Qwen3OmniClient,
        read_audio_file,
    )
except ImportError:
    from qwen3_omni_client import (  # type: ignore[no-redef]
        ASRConfig,
        ClientConfig,
        Qwen3ASRClient,
        Qwen3OmniClient,
        read_audio_file,
    )


@dataclass(frozen=True)
class LatencySample:
    run: int
    asr_seconds: float
    asr_realtime_factor: float | None
    llm_ttft_seconds: float
    llm_total_seconds: float
    pipeline_first_text_seconds: float
    pipeline_total_seconds: float
    transcript: str
    translation: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the current C500 ASR -> streaming LLM latency using the "
            "OpenAI-compatible APIs documented in C500模型服务API交付文档v2."
        )
    )
    parser.add_argument("audio_file", type=Path)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--base-url", default="http://172.26.63.11:8003/v1")
    parser.add_argument("--model", default="qwen3-omni")
    parser.add_argument("--asr-base-url", default="http://172.26.63.11:8004/v1")
    parser.add_argument("--asr-model", default="qwen3-asr")
    parser.add_argument("--source-language", default="Chinese")
    parser.add_argument("--target-language", default="English")
    parser.add_argument("--max-tokens", type=int, default=128)
    return parser


def wav_duration_seconds(path: Path) -> float | None:
    if path.suffix.lower() != ".wav":
        return None
    with wave.open(str(path), "rb") as audio:
        return audio.getnframes() / audio.getframerate()


def percentile(values: list[float], percentile_value: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile_value * len(ordered)) - 1)
    return ordered[index]


def metric_summary(samples: list[LatencySample], field: str) -> dict[str, float]:
    values = [
        float(value)
        for sample in samples
        if (value := getattr(sample, field)) is not None
    ]
    if not values:
        raise ValueError(f"metric {field!r} has no numeric samples")
    return {
        "median": statistics.median(values),
        "p95_nearest_rank": percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


async def measure_once(
    *,
    run: int,
    audio_bytes: bytes,
    audio_format: str,
    audio_duration: float | None,
    asr: Qwen3ASRClient,
    llm: Qwen3OmniClient,
) -> LatencySample:
    asr_started = time.perf_counter()
    transcript = await asr.transcribe_audio(
        audio_bytes,
        audio_format=audio_format,
    )
    asr_seconds = time.perf_counter() - asr_started
    if not transcript:
        raise RuntimeError("ASR returned an empty transcript")

    llm_started = time.perf_counter()
    first_delta_at: float | None = None

    def on_delta(_: str) -> None:
        nonlocal first_delta_at
        if first_delta_at is None:
            first_delta_at = time.perf_counter()

    translation = await llm.translate_text(transcript, on_delta=on_delta)
    llm_finished = time.perf_counter()
    if first_delta_at is None:
        raise RuntimeError("LLM stream returned no text delta")

    llm_ttft_seconds = first_delta_at - llm_started
    llm_total_seconds = llm_finished - llm_started
    return LatencySample(
        run=run,
        asr_seconds=asr_seconds,
        asr_realtime_factor=(
            asr_seconds / audio_duration if audio_duration is not None else None
        ),
        llm_ttft_seconds=llm_ttft_seconds,
        llm_total_seconds=llm_total_seconds,
        pipeline_first_text_seconds=asr_seconds + llm_ttft_seconds,
        pipeline_total_seconds=asr_seconds + llm_total_seconds,
        transcript=transcript,
        translation=translation,
    )


async def run(args: argparse.Namespace) -> dict[str, object]:
    if args.runs < 1 or args.warmup_runs < 0:
        raise ValueError("runs must be positive and warmup-runs cannot be negative")

    audio_bytes, audio_format = read_audio_file(args.audio_file)
    audio_duration = wav_duration_seconds(args.audio_file)
    llm_config = ClientConfig(
        base_url=args.base_url,
        model=args.model,
        source_language=args.source_language,
        target_language=args.target_language,
        max_tokens=args.max_tokens,
        temperature=0.1,
    )
    asr_config = ASRConfig(
        base_url=args.asr_base_url,
        model=args.asr_model,
    )

    async with (
        Qwen3OmniClient(llm_config) as llm,
        Qwen3ASRClient(asr_config) as asr,
    ):
        llm_status = await llm.check_server()
        asr_status = await asr.check_server()
        if not llm_status["configured_model_available"]:
            raise RuntimeError(f"LLM model {args.model!r} is not available")
        if not asr_status["configured_model_available"]:
            raise RuntimeError(f"ASR model {args.asr_model!r} is not available")

        for warmup in range(args.warmup_runs):
            print(f"warm-up {warmup + 1}/{args.warmup_runs}", flush=True)
            await measure_once(
                run=0,
                audio_bytes=audio_bytes,
                audio_format=audio_format,
                audio_duration=audio_duration,
                asr=asr,
                llm=llm,
            )

        samples: list[LatencySample] = []
        for run_number in range(1, args.runs + 1):
            sample = await measure_once(
                run=run_number,
                audio_bytes=audio_bytes,
                audio_format=audio_format,
                audio_duration=audio_duration,
                asr=asr,
                llm=llm,
            )
            samples.append(sample)
            print(
                f"run {run_number}/{args.runs}: ASR={sample.asr_seconds:.3f}s, "
                f"LLM TTFT={sample.llm_ttft_seconds:.3f}s, "
                f"first translated text={sample.pipeline_first_text_seconds:.3f}s, "
                f"pipeline total={sample.pipeline_total_seconds:.3f}s",
                flush=True,
            )

    metric_names = [
        "asr_seconds",
        "asr_realtime_factor",
        "llm_ttft_seconds",
        "llm_total_seconds",
        "pipeline_first_text_seconds",
        "pipeline_total_seconds",
    ]
    result: dict[str, object] = {
        "measured_at_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "audio_file": str(args.audio_file.resolve()),
        "audio_format": audio_format,
        "audio_duration_seconds": audio_duration,
        "runs": args.runs,
        "warmup_runs": args.warmup_runs,
        "llm": {"base_url": args.base_url, "model": args.model},
        "asr": {"base_url": args.asr_base_url, "model": args.asr_model},
        "summary": {
            name: metric_summary(samples, name) for name in metric_names
        },
        "samples": [asdict(sample) for sample in samples],
    }
    return result


def main() -> int:
    args = build_parser().parse_args()
    result = asyncio.run(run(args))
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
