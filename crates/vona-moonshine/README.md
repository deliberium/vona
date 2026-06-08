# vona-moonshine

Protected Moonshine local STT adapter for Vona.

`NativeMoonshineTranscriber` implements Vona's `AudioTranscriber` trait by
loading `libmoonshine.dylib` directly through Moonshine's C API.
`ProtectedMoonshineTranscriber` remains available as a contained worker fallback:
the parent sends a JSON-lines request header followed by raw little-endian `f32`
PCM samples; the worker keeps the Moonshine model loaded and returns one
transcript response per request.

## Configuration

- `VONA_MOONSHINE_PYTHON`: Python executable with `moonshine-voice` installed.
- `VONA_MOONSHINE_WORKER_SCRIPT`: override the bundled worker script path.
- `VONA_MOONSHINE_LIBRARY_PATH`: `libmoonshine.dylib` path for native mode.
- `VONA_MOONSHINE_MODEL_PATH`: downloaded Moonshine model directory for native mode.
- `VONA_MOONSHINE_CACHE_ROOT`: Moonshine model cache root.
- `VONA_MOONSHINE_ARCH`: Moonshine model arch, default `MEDIUM_STREAMING`.
- `VONA_MOONSHINE_HOTWORDS`: comma-separated `replacement=variant|variant` entries.

The worker expects 16 kHz mono audio. For the current Deliberium/Vona voice
profile, `MEDIUM_STREAMING` is the recommended local STT candidate: the labeled
synthetic benchmark reported avg WER `0.019`, avg STT `392.1 ms`, and near-exact
transcripts for `11/12` cases.

Native smoke test:

```bash
VONA_MOONSHINE_LIBRARY_PATH=/path/to/libmoonshine.dylib \
VONA_MOONSHINE_MODEL_PATH=/path/to/medium-streaming-en/quantized \
VONA_MOONSHINE_SMOKE_PCM16LE=/path/to/audio.pcm16le \
cargo run -p vona-moonshine --example native_smoke
```
