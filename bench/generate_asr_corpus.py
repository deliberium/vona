#!/usr/bin/env python3
"""Generate a deterministic labelled ASR corpus as raw 16 kHz mono PCM.

The corpus is synthetic by design: it is useful for repeatable regression and
latency testing, but it must not be represented as human-recorded evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import tempfile
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


CATEGORIES: dict[str, list[str]] = {
    "short_command": [
        "Start listening.",
        "Pause the session.",
        "Open the dashboard.",
        "Mute the microphone.",
        "Cancel that request.",
        "Read the summary.",
        "Save this note.",
        "Resume playback.",
    ],
    "simple_question": [
        "What day is it today?",
        "What time is the next meeting?",
        "Can you explain that again?",
        "What is the weather like tomorrow?",
        "Who was Albert Einstein?",
        "What does local first mean?",
        "How do I restart the service?",
        "Where did the benchmark report go?",
    ],
    "long_form": [
        "Give me a careful summary of the architecture tradeoffs, including latency, accuracy, privacy, and operational cost.",
        "Draft a concise project update for the team that explains what changed, what passed testing, and what remains risky.",
        "Walk me through the decision tree for choosing a local speech model, a cloud fallback, and a streaming text to speech provider.",
        "Explain how the voice pipeline should behave when the user interrupts halfway through a long answer.",
        "Prepare a release note that mentions native speech recognition, configurable hotwords, and benchmark evidence.",
    ],
    "complex_reasoning": [
        "If local recognition is confident but intent plausibility is low, what should the runtime do before answering?",
        "Compare the privacy benefit of local speech recognition with the reliability benefit of a cloud fallback.",
        "If first audio latency improves but total answer latency gets worse, how should the product policy decide?",
        "When a wake gate opens and the next utterance is ambiguous, describe the safest conversational behavior.",
        "Explain why benchmark evidence should separate synthetic generated audio from human recorded audio.",
    ],
    "technical_debug": [
        "The speech worker is loaded but transcripts are empty, list three likely causes.",
        "The local model cache exists but the loader still fetches from the network, what should I check?",
        "The microphone level is high but recognition confidence is low, suggest a debugging sequence.",
        "The text generator streams quickly but audio starts late, identify the bottleneck candidates.",
        "The wake gate fires but the first user turn is missed, explain the instrumentation we need.",
    ],
    "numbers_dates": [
        "Schedule the review for June sixth two thousand twenty six at nine thirty.",
        "Read back the ticket number V O N A dash four zero eight seven.",
        "The target latency is under eight hundred milliseconds for the first audio chunk.",
        "Compare model version zero point two point zero with zero point one point one.",
        "Set the retry count to three and the timeout to forty five seconds.",
    ],
    "proper_nouns": [
        "Summarize the difference between Moonshine, Whisper, Deepgram, and Kokoro.",
        "Explain how Vona connects local speech recognition with Ollama text generation.",
        "Tell me why Gemma and Phi four mini might be routed differently.",
        "Normalize the phrase Vona wake when it appears in a transcript.",
        "Mention Apple Silicon, Metal, and MLX in the same answer.",
    ],
    "punctuation_heavy": [
        "First check readiness, then summarize results, then ask one follow up question.",
        "Say yes if the worker is healthy, otherwise say no and explain why.",
        "Create three bullets: latency, accuracy, and operational risk.",
        "Read this exactly as a spoken sentence, not as punctuation symbols.",
        "If the score is low, retry once; if it is still low, ask for clarification.",
    ],
    "conversational_fragment": [
        "Actually, make that shorter.",
        "No, I meant the local path.",
        "That sounds right, keep going.",
        "Wait, cancel the previous instruction.",
        "Can you say that in plain English?",
        "One more thing about the benchmark.",
        "Not that one, the other adapter.",
        "Let's try the faster policy path.",
    ],
    "real_world_task": [
        "Write a polite response saying I am running late but will join the call in ten minutes.",
        "Summarize the last decision and turn it into a checklist for tomorrow morning.",
        "Create a short incident note explaining that speech recognition failed because the model was unavailable.",
        "Draft a message asking the team to record ten clean human corpus examples.",
        "Turn this into an action item with an owner, a deadline, and a verification step.",
    ],
}


QUALIFIERS = [
    "Keep the answer concise.",
    "Use a friendly tone.",
    "Do not mention private project names.",
    "Focus on release readiness.",
    "Include the main risk.",
    "Prefer local processing where practical.",
    "Make it useful for a technical reader.",
    "Avoid marketing fluff.",
]

DOMAINS = [
    "voice runtime",
    "release benchmark",
    "local model cache",
    "wake gate",
    "speech worker",
    "mobile navigation",
    "assistant policy",
    "offline fallback",
    "audio transport",
    "quality ladder",
]

NAMES = [
    "Ada",
    "Grace",
    "Alan",
    "Katherine",
    "Linus",
    "Margaret",
    "Barbara",
    "Dennis",
    "Radia",
    "Claude",
]

OBJECTS = [
    "incident note",
    "standup summary",
    "architecture review",
    "customer reply",
    "test report",
    "handoff memo",
    "launch checklist",
    "debug trace",
    "privacy assessment",
    "meeting brief",
]

LOCATIONS = [
    "London",
    "Manchester",
    "Edinburgh",
    "Cardiff",
    "Bristol",
    "Dublin",
    "New York",
    "Toronto",
    "Berlin",
    "Paris",
]


def variation(index: int, rng: random.Random, category: str) -> str:
    name = rng.choice(NAMES)
    domain = rng.choice(DOMAINS)
    obj = rng.choice(OBJECTS)
    location = rng.choice(LOCATIONS)
    ticket = f"VONA-{1000 + index}"
    minute = 5 + (index % 50)
    day = 1 + (index % 27)
    hour = 8 + (index % 10)
    if category == "numbers_dates":
        return f"Reference {ticket}, June {day}, {hour}:{minute:02d}, {location}."
    if category == "proper_nouns":
        return f"Use {name}, {location}, {ticket}, and {domain}."
    if category == "short_command":
        return f"For {obj} {ticket}."
    if category == "conversational_fragment":
        return f"I mean {domain}."
    if category == "simple_question":
        return f"For {name} in {location}."
    if category == "punctuation_heavy":
        return f"Use {ticket}: {domain}, {obj}, {location}."
    if category == "real_world_task":
        return f"For {name}, include {ticket}."
    if category == "technical_debug":
        return f"Component {domain}, trace {ticket}."
    if category == "complex_reasoning":
        return f"Use {domain} for {name}."
    return f"Use {obj}, {domain}, {location}."


def build_cases(count: int, seed: int, source_type: str) -> list[dict[str, object]]:
    rng = random.Random(seed)
    categories = sorted(CATEGORIES)
    cases: list[dict[str, object]] = []
    for index in range(count):
        category = categories[index % len(categories)]
        base = rng.choice(CATEGORIES[category])
        detail = variation(index + 1, rng, category)
        if category in {"long_form", "complex_reasoning", "technical_debug", "real_world_task"}:
            text = f"{base} {detail}"
        elif index % 11 == 0:
            text = f"{base} {detail} {rng.choice(QUALIFIERS)}"
        else:
            text = f"{base} {detail}"
        case_id = f"case-{index + 1:04d}"
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        cases.append(
            {
                "id": case_id,
                "category": category,
                "expected": text,
                "sample_rate_hz": 16000,
                "channels": 1,
                "source_type": source_type,
                "text_sha256_16": text_hash,
            }
        )
    return cases


def synthesize_case(case: dict[str, object], audio_dir: Path, voice: str | None, force: bool) -> None:
    pcm_path = audio_dir / f"{case['id']}.pcm16le"
    case["audio_path"] = str(Path("audio") / pcm_path.name)
    if pcm_path.exists() and not force:
        return

    with tempfile.TemporaryDirectory(prefix="vona-asr-corpus-") as tmp:
        aiff_path = Path(tmp) / "spoken.aiff"
        wav_path = Path(tmp) / "spoken.wav"
        say_cmd = ["/usr/bin/say", "-o", str(aiff_path)]
        if voice:
            say_cmd.extend(["-v", voice])
        say_cmd.append(str(case["expected"]))
        subprocess.run(say_cmd, check=True)
        subprocess.run(
            [
                "/usr/bin/afconvert",
                "-f",
                "WAVE",
                "-d",
                "LEI16@16000",
                "-c",
                "1",
                str(aiff_path),
                str(wav_path),
            ],
            check=True,
        )
        with wave.open(str(wav_path), "rb") as wav:
            if wav.getframerate() != 16000 or wav.getnchannels() != 1 or wav.getsampwidth() != 2:
                raise RuntimeError(
                    f"unexpected wav format for {case['id']}: "
                    f"{wav.getframerate()} Hz, {wav.getnchannels()} channels, "
                    f"{wav.getsampwidth()} bytes"
                )
            pcm_path.write_bytes(wav.readframes(wav.getnframes()))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--voice", default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--source-type", default="generated-macos-say")
    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("--count must be positive")

    out_dir = args.out_dir
    audio_dir = out_dir / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    cases = build_cases(args.count, args.seed, args.source_type)
    voice = args.voice.strip() or None

    for case in cases:
        case["audio_path"] = str(Path("audio") / f"{case['id']}.pcm16le")

    if not args.manifest_only:
        jobs = max(1, args.jobs)
        completed = 0
        if jobs == 1:
            for index, case in enumerate(cases, 1):
                synthesize_case(case, audio_dir, voice, args.force)
                completed = index
                if completed == 1 or completed % 100 == 0 or completed == len(cases):
                    print(f"generated {completed}/{len(cases)}", flush=True)
        else:
            with ThreadPoolExecutor(max_workers=jobs) as executor:
                futures = [
                    executor.submit(synthesize_case, case, audio_dir, voice, args.force)
                    for case in cases
                ]
                for future in as_completed(futures):
                    future.result()
                    completed += 1
                    if completed == 1 or completed % 100 == 0 or completed == len(cases):
                        print(f"generated {completed}/{len(cases)}", flush=True)

    manifest_path = out_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, sort_keys=True) + "\n")

    summary = {
        "cases": len(cases),
        "seed": args.seed,
        "source_type": args.source_type,
        "voice": voice or "system-default",
        "manifest": str(manifest_path),
        "categories": {category: 0 for category in sorted(CATEGORIES)},
    }
    for case in cases:
        summary["categories"][str(case["category"])] += 1
    (out_dir / "corpus-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
