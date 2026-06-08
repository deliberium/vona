#!/usr/bin/env python3
import argparse
import json
import queue
import sys
import threading
import traceback

from mlx_vlm import load
from mlx_vlm.generate import apply_chat_template, stream_generate


def write(payload):
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def start_reader():
    messages = queue.Queue()

    def read_loop():
        for line in sys.stdin:
            try:
                messages.put(json.loads(line))
            except Exception as error:
                messages.put({"error": str(error)})
        messages.put({"eof": True})

    thread = threading.Thread(target=read_loop, daemon=True)
    thread.start()
    return messages


def take_next(messages, pending):
    if pending:
        return pending.pop(0)
    return messages.get()


def drain_control_messages(messages, pending, active_id):
    while True:
        try:
            message = messages.get_nowait()
        except queue.Empty:
            return False

        if message.get("cancel") and message.get("id") == active_id:
            return True
        pending.append(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--temperature", default="0.0")
    args = parser.parse_args()

    model, processor = load(args.model)
    config = model.config
    messages = start_reader()
    pending = []
    write({"ready": True, "model": args.model})

    while True:
        request = take_next(messages, pending)
        if request.get("eof"):
            break
        if request.get("cancel"):
            continue
        try:
            request_id = request.get("id")
            prompt = request["prompt"]
            max_tokens = int(request.get("max_tokens", args.max_tokens))
            temperature = float(request.get("temperature", args.temperature))
            prompt = apply_chat_template(
                processor,
                config,
                prompt,
                num_images=0,
                num_audios=0,
                enable_thinking=False,
            )
            final_result = None
            canceled = False
            for result in stream_generate(
                model,
                processor,
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                verbose=False,
            ):
                if drain_control_messages(messages, pending, request_id):
                    canceled = True
                    break
                final_result = result
                if result.text:
                    write({"id": request_id, "delta": result.text})
            write(
                {
                    "id": request_id,
                    "done": True,
                    "canceled": canceled,
                    "prompt_tokens": getattr(final_result, "prompt_tokens", 0),
                    "generation_tokens": getattr(final_result, "generation_tokens", 0),
                    "prompt_tps": getattr(final_result, "prompt_tps", 0.0),
                    "generation_tps": getattr(final_result, "generation_tps", 0.0),
                    "peak_memory_gb": getattr(final_result, "peak_memory", 0.0),
                    "finish_reason": getattr(final_result, "finish_reason", None),
                }
            )
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            write({"id": request.get("id"), "error": str(error)})


if __name__ == "__main__":
    main()
