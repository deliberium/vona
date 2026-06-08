#!/usr/bin/env python3
"""Persistent Moonshine STT worker for Vona.

Protocol:
  stdin: JSON header line with id/sample_rate_hz/channels/samples, then f32le samples.
  stdout: one JSON response line with id/transcript/error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from moonshine_voice.download import get_model_for_language
from moonshine_voice.transcriber import ModelArch, Transcriber


def read_exact(size: int) -> bytes:
    data = sys.stdin.buffer.read(size)
    if len(data) != size:
        raise EOFError(f"expected {size} bytes, got {len(data)}")
    return data


def transcript_text(transcript: object) -> str:
    lines = getattr(transcript, "lines", None)
    if lines is not None:
        return " ".join(getattr(line, "text", "") for line in lines).strip()
    return str(transcript).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="MEDIUM_STREAMING")
    parser.add_argument("--cache-root")
    args = parser.parse_args()

    arch = getattr(ModelArch, args.arch)
    cache_root = Path(args.cache_root) if args.cache_root else None
    model_path, resolved_arch = get_model_for_language("en", arch, cache_root=cache_root)
    transcriber = Transcriber(model_path, resolved_arch)
    print(
        json.dumps({"ready": True, "arch": resolved_arch.name, "model_path": str(model_path)}),
        flush=True,
    )

    while True:
        header = sys.stdin.buffer.readline()
        if not header:
            return 0
        try:
            request = json.loads(header.decode("utf-8"))
            sample_count = int(request["samples"])
            sample_rate_hz = int(request["sample_rate_hz"])
            channels = int(request["channels"])
            raw = read_exact(sample_count * 4)
            samples = np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=False)
            if channels != 1:
                raise ValueError(f"Moonshine expects mono audio, got {channels} channels")
            transcript = transcriber.transcribe_without_streaming(
                samples.tolist(),
                sample_rate=sample_rate_hz,
                flags=0,
            )
            response = {
                "id": request["id"],
                "transcript": transcript_text(transcript),
                "error": None,
            }
        except Exception as exc:
            response = {
                "id": request.get("id") if "request" in locals() and isinstance(request, dict) else 0,
                "transcript": None,
                "error": str(exc),
            }
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
