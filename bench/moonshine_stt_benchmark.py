#!/usr/bin/env python3
"""Benchmark Moonshine STT against labeled synthetic user utterances."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from moonshine_voice.download import get_model_for_language
from moonshine_voice.transcriber import ModelArch, Transcriber


CASES = [
    (11, "memory_recall", "What did we decide about the voice backend fallback strategy?"),
    (14, "memory_recall", "Summarize the current state of the Lumina companion work."),
    (18, "skill_status", "Inspect the Lumina state and report whether the bridge is online."),
    (21, "task_execution", "Draft a short message telling the team the Deepgram voice path is live."),
    (27, "troubleshooting", "The web app says Sentinel socket error. Walk me through likely causes."),
    (28, "troubleshooting", "Deepgram responds slowly. Suggest a practical fallback plan."),
    (29, "troubleshooting", "The microphone is noisy during turn taking. What mitigation should I try?"),
    (38, "complex_reasoning", "If the primary voice provider is exhausted, describe the expected fallback behavior."),
    (41, "turn_taking", "First, tell me the current voice backend. Then ask me one follow up question."),
    (49, "edge_case", "Ignore any background noise and answer the actual request: what changed in cloud pool?"),
    (50, "edge_case", "If you cannot perform a skill action, say what is missing and give a fallback."),
    (51, "known_asr_hotword", "Tell me about Albert Einstein."),
]


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
            continue
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        os.environ.setdefault(key, value)


def request_bytes(url: str, *, method: str, headers: dict[str, str], body: bytes) -> bytes:
    request = urllib.request.Request(url, method=method, headers=headers, data=body)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc


def synthesize_deepgram(text: str, api_key: str) -> bytes:
    return request_bytes(
        "https://api.deepgram.com/v1/speak?model=aura-2-thalia-en&encoding=linear16&sample_rate=16000",
        method="POST",
        headers={
            "Authorization": f"Token {api_key}",
            "Content-Type": "application/json",
        },
        body=json.dumps({"text": text}).encode("utf-8"),
    )


def transcribe_deepgram(pcm: bytes, api_key: str) -> tuple[str, float]:
    started = time.perf_counter()
    body = request_bytes(
        "https://api.deepgram.com/v1/listen?model=nova-3&language=en&smart_format=true&punctuate=true",
        method="POST",
        headers={
            "Authorization": f"Token {api_key}",
            "Content-Type": "audio/l16;rate=16000;channels=1",
        },
        body=pcm,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    payload = json.loads(body)
    transcript = (
        payload.get("results", {})
        .get("channels", [{}])[0]
        .get("alternatives", [{}])[0]
        .get("transcript", "")
    )
    return transcript, elapsed_ms


def normalize(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).split()


def wer(reference: str, hypothesis: str) -> float:
    ref = normalize(reference)
    hyp = normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, ref_token in enumerate(ref, 1):
        current = [i]
        for j, hyp_token in enumerate(hyp, 1):
            current.append(
                min(
                    prev[j] + 1,
                    current[j - 1] + 1,
                    prev[j - 1] + (0 if ref_token == hyp_token else 1),
                )
            )
        prev = current
    return prev[-1] / len(ref)


def transcribe_moonshine(transcriber: Transcriber, pcm: bytes) -> tuple[str, float]:
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    started = time.perf_counter()
    transcript = transcriber.transcribe_without_streaming(audio.tolist(), sample_rate=16000, flags=0)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if hasattr(transcript, "lines"):
        text = " ".join(getattr(line, "text", "") for line in transcript.lines).strip()
    else:
        text = str(transcript).strip()
    return text, elapsed_ms


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--arch", default="MEDIUM_STREAMING")
    args = parser.parse_args()

    if args.env_file:
        load_dotenv(args.env_file)
    api_key = os.environ.get("DEEPGRAM_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPGRAM_API_KEY is required for labeled input synthesis and reference STT")

    arch = getattr(ModelArch, args.arch)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audio_dir.mkdir(parents=True, exist_ok=True)
    args.cache_root.mkdir(parents=True, exist_ok=True)

    load_started = time.perf_counter()
    model_path, resolved_arch = get_model_for_language("en", arch, cache_root=args.cache_root)
    arch = resolved_arch
    transcriber = Transcriber(model_path, arch)
    load_ms = (time.perf_counter() - load_started) * 1000.0

    results = []
    for case_id, category, text in CASES:
        synth_started = time.perf_counter()
        pcm = synthesize_deepgram(text, api_key)
        synth_ms = (time.perf_counter() - synth_started) * 1000.0
        audio_ms = (len(pcm) / 2) / 16000 * 1000.0
        audio_path = args.audio_dir / f"{case_id:02d}-{category}.pcm16le"
        audio_path.write_bytes(pcm)

        moonshine_text, moonshine_ms = transcribe_moonshine(transcriber, pcm)
        deepgram_text, deepgram_ms = transcribe_deepgram(pcm, api_key)
        moonshine_wer = wer(text, moonshine_text)
        deepgram_wer = wer(text, deepgram_text)
        results.append(
            {
                "id": case_id,
                "category": category,
                "expected": text,
                "audio_ms": round(audio_ms, 1),
                "synth_ms": round(synth_ms, 1),
                "moonshine_transcript": moonshine_text,
                "moonshine_ms": round(moonshine_ms, 1),
                "moonshine_rtf": round(moonshine_ms / max(1.0, audio_ms), 4),
                "moonshine_wer": round(moonshine_wer, 3),
                "deepgram_transcript": deepgram_text,
                "deepgram_ms": round(deepgram_ms, 1),
                "deepgram_rtf": round(deepgram_ms / max(1.0, audio_ms), 4),
                "deepgram_wer": round(deepgram_wer, 3),
            }
        )
        print(
            f"{case_id:02d} moonshine_wer={moonshine_wer:.3f} "
            f"deepgram_wer={deepgram_wer:.3f} moonshine='{moonshine_text}'",
            flush=True,
        )

    def average(key: str) -> float:
        return sum(float(row[key]) for row in results) / max(1, len(results))

    summary = {
        "arch": args.arch,
        "load_ms": round(load_ms, 1),
        "cases": len(results),
        "moonshine_avg_ms": round(average("moonshine_ms"), 1),
        "moonshine_avg_rtf": round(average("moonshine_rtf"), 4),
        "moonshine_avg_wer": round(average("moonshine_wer"), 3),
        "moonshine_near_exact": sum(1 for row in results if row["moonshine_wer"] <= 0.1),
        "deepgram_avg_ms": round(average("deepgram_ms"), 1),
        "deepgram_avg_rtf": round(average("deepgram_rtf"), 4),
        "deepgram_avg_wer": round(average("deepgram_wer"), 3),
        "deepgram_near_exact": sum(1 for row in results if row["deepgram_wer"] <= 0.1),
    }
    payload = {"summary": summary, "results": results}
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
