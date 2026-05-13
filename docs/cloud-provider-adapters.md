# Cloud Provider Adapters

This document records the current provider split used by Vona as of May 13, 2026.

## Research Snapshot

Official provider documentation currently points to three different shapes:

- OpenAI Realtime exposes bidirectional audio events such as `input_audio_buffer.append` and `response.output_audio.delta`, which fits `RealtimeVoiceBackend`.
- Gemini Live exposes a server-to-server WebSocket Live API with realtime audio input and inline audio output, which fits `RealtimeVoiceBackend`.
- Azure now has a Voice Live realtime endpoint, while Azure Speech still exposes mature STT/TTS surfaces for cascaded backends.
- ElevenLabs documents WebSocket text-to-speech streaming through `/v1/text-to-speech/{voice_id}/stream-input`, which is a TTS component rather than native STS.
- Deepgram documents Flux/listen streaming STT and Aura streaming TTS. Those are speech components for cascaded voice agents.

Sources:

- [OpenAI Realtime model capabilities](https://platform.openai.com/docs/guides/realtime-model-capabilities)
- [OpenAI Realtime client events](https://platform.openai.com/docs/api-reference/realtime-beta-client-events/response?api-mode=chat)
- [Gemini Live API guide](https://ai.google.dev/gemini-api/docs/live-guide)
- [Gemini Live API get started](https://ai.google.dev/gemini-api/docs/live)
- [Azure Voice Live API](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/voice-live-how-to)
- [Azure Speech to text](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/how-to-recognize-speech)
- [Azure Text to speech REST API](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/rest-text-to-speech)
- [ElevenLabs WebSocket TTS](https://elevenlabs.io/docs/api-reference/websocket)
- [Deepgram Streaming TTS](https://developers.deepgram.com/docs/streaming-text-to-speech)
- [Deepgram Live Audio reference](https://developers.deepgram.com/reference/speech-to-text/listen-streaming)
- [Deepgram Flux streaming audio](https://developers.deepgram.com/speech-to-text/streaming-audio)

## Crates

| Crate | Role | Live Network Transport |
|-------|------|------------------------|
| `vona-openai-realtime` | OpenAI Realtime config and event mapping | Not run in CI |
| `vona-gemini-live` | Gemini Live config and event mapping | Not run in CI |
| `vona-azure-speech` | Azure Voice Live endpoint config plus Speech STT/TTS helpers | Not run in CI |
| `vona-elevenlabs` | ElevenLabs streaming TTS helpers | Not run in CI |
| `vona-deepgram` | Deepgram streaming STT/TTS helpers | Not run in CI |

Provider crates deliberately keep deterministic tests at the protocol boundary. Credentialed tests should be added as ignored integration tests once each provider account, spend limit, and data-retention policy is explicit.

## Audio Format

Vona core audio frames use normalized `f32` samples. Provider crates that send PCM over JSON/WebSocket convert to 16-bit little-endian PCM and base64-encode those bytes where required by provider docs.

## Packaging

The `vona` umbrella crate exposes cloud adapters through:

- `openai-realtime`
- `gemini-live`
- `azure-speech`
- `elevenlabs`
- `deepgram`
- `cloud`
- `all`

Applications that want a small dependency graph should depend on the provider crate directly.
