# MLX + Ollama Voice E2E Benchmark

| Metric | Value |
|---|---:|
| Cases | 100 |
| Non-empty transcripts | 0 |
| Qwen3 TTS load | 45235 ms |
| Qwen3 TTS weights | 478 |
| Qwen3 vocoder weights | 496 |
| Whisper load | 65798 ms |
| Whisper weights | 1258 |
| Min audio tokens | 4 |
| Max audio tokens | 4 |
| Max decode tokens | 8 |
| Total measured run | 438979 ms |
| TTS avg | 1266.26 ms |
| STT avg | 1041.00 ms |
| Ollama avg | 2080.79 ms |

> Historical validation finding: this run used an earlier Qwen3 vocoder stride inference bug that decoded 4x too few samples per codec frame, so the empty transcripts below should not be treated as current TTS quality evidence. The stride bug has since been fixed and validated with independent `whisper-cli` ASR: `Hello from native Vona MLX.` decoded to 67,200 samples / 2.8 s and transcribed as “Hello from native Vona MLX.”; `The quick brown fox jumps over the lazy dog.` decoded to 78,720 samples / 3.28 s and transcribed exactly. The 100-case table remains useful as the pre-fix throughput/error-handling record.

| # | TTS ms | STT ms | Ollama ms | Samples | Peak | Frames | Response chars | Prompt | Transcript |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 1332 | 1030 | 4165 | 1920 | 1.0000 | 129 | 625 | Hello from native Vona MLX case 1 |  |
| 2 | 1533 | 1427 | 2003 | 1920 | 1.0000 | 129 | 625 | Local voice test for Ollama case 2 |  |
| 3 | 1248 | 1033 | 2001 | 1920 | 0.8654 | 129 | 625 | Rust audio pipeline ready case 3 |  |
| 4 | 1271 | 1034 | 1995 | 1920 | 1.0000 | 129 | 625 | Apple Silicon speech check case 4 |  |
| 5 | 1262 | 1040 | 2007 | 1920 | 1.0000 | 129 | 625 | Streaming adapter test case 5 |  |
| 6 | 1268 | 1034 | 2009 | 1920 | 1.0000 | 129 | 625 | Whisper transcription pass case 6 |  |
| 7 | 1189 | 1045 | 1997 | 1920 | 0.9980 | 129 | 625 | Qwen speech synthesis pass case 7 |  |
| 8 | 1232 | 1044 | 1994 | 1920 | 1.0000 | 129 | 625 | Benchmark sample complete case 8 |  |
| 9 | 1260 | 1037 | 1998 | 1920 | 1.0000 | 129 | 625 | Realtime voice path active case 9 |  |
| 10 | 1252 | 1039 | 1994 | 1920 | 1.0000 | 129 | 625 | Vona local inference check case 10 |  |
| 11 | 1249 | 1031 | 1988 | 1920 | 0.2982 | 129 | 625 | Hello from native Vona MLX case 11 |  |
| 12 | 1271 | 1037 | 2015 | 1920 | 1.0000 | 129 | 625 | Local voice test for Ollama case 12 |  |
| 13 | 1253 | 1035 | 1994 | 1920 | 1.0000 | 129 | 625 | Rust audio pipeline ready case 13 |  |
| 14 | 1262 | 1036 | 2011 | 1920 | 1.0000 | 129 | 625 | Apple Silicon speech check case 14 |  |
| 15 | 1260 | 1036 | 1992 | 1920 | 1.0000 | 129 | 625 | Streaming adapter test case 15 |  |
| 16 | 1261 | 1035 | 2007 | 1920 | 1.0000 | 129 | 625 | Whisper transcription pass case 16 |  |
| 17 | 1258 | 1022 | 1969 | 1920 | 0.5022 | 129 | 625 | Qwen speech synthesis pass case 17 |  |
| 18 | 1263 | 1038 | 2007 | 1920 | 1.0000 | 129 | 625 | Benchmark sample complete case 18 |  |
| 19 | 1254 | 1028 | 1998 | 1920 | 1.0000 | 129 | 625 | Realtime voice path active case 19 |  |
| 20 | 1256 | 1038 | 2002 | 1920 | 1.0000 | 129 | 625 | Vona local inference check case 20 |  |
| 21 | 1265 | 1031 | 2024 | 1920 | 0.8452 | 129 | 625 | Hello from native Vona MLX case 21 |  |
| 22 | 1276 | 1062 | 2009 | 1920 | 1.0000 | 129 | 625 | Local voice test for Ollama case 22 |  |
| 23 | 1270 | 1039 | 2014 | 1920 | 1.0000 | 129 | 625 | Rust audio pipeline ready case 23 |  |
| 24 | 1262 | 1041 | 2011 | 1920 | 1.0000 | 129 | 625 | Apple Silicon speech check case 24 |  |
| 25 | 1246 | 1040 | 2002 | 1920 | 1.0000 | 129 | 625 | Streaming adapter test case 25 |  |
| 26 | 1258 | 1036 | 2000 | 1920 | 1.0000 | 129 | 625 | Whisper transcription pass case 26 |  |
| 27 | 1254 | 1043 | 2004 | 1920 | 0.7294 | 129 | 625 | Qwen speech synthesis pass case 27 |  |
| 28 | 1269 | 1042 | 2005 | 1920 | 1.0000 | 129 | 625 | Benchmark sample complete case 28 |  |
| 29 | 1254 | 1037 | 2006 | 1920 | 1.0000 | 129 | 625 | Realtime voice path active case 29 |  |
| 30 | 1267 | 1044 | 2008 | 1920 | 1.0000 | 129 | 625 | Vona local inference check case 30 |  |
| 31 | 1272 | 1049 | 2019 | 1920 | 0.4761 | 129 | 625 | Hello from native Vona MLX case 31 |  |
| 32 | 1263 | 1040 | 2017 | 1920 | 0.8513 | 129 | 625 | Local voice test for Ollama case 32 |  |
| 33 | 1270 | 1023 | 2008 | 1920 | 1.0000 | 129 | 625 | Rust audio pipeline ready case 33 |  |
| 34 | 1256 | 1024 | 2017 | 1920 | 1.0000 | 129 | 625 | Apple Silicon speech check case 34 |  |
| 35 | 1258 | 1036 | 2009 | 1920 | 1.0000 | 129 | 625 | Streaming adapter test case 35 |  |
| 36 | 1270 | 1035 | 2006 | 1920 | 1.0000 | 129 | 625 | Whisper transcription pass case 36 |  |
| 37 | 1274 | 1038 | 2008 | 1920 | 0.5050 | 129 | 625 | Qwen speech synthesis pass case 37 |  |
| 38 | 1261 | 1039 | 2007 | 1920 | 1.0000 | 129 | 625 | Benchmark sample complete case 38 |  |
| 39 | 1259 | 1037 | 2005 | 1920 | 1.0000 | 129 | 625 | Realtime voice path active case 39 |  |
| 40 | 1264 | 1038 | 2008 | 1920 | 1.0000 | 129 | 625 | Vona local inference check case 40 |  |
| 41 | 1261 | 1026 | 2009 | 1920 | 0.6255 | 129 | 625 | Hello from native Vona MLX case 41 |  |
| 42 | 1262 | 1037 | 2018 | 1920 | 0.8307 | 129 | 625 | Local voice test for Ollama case 42 |  |
| 43 | 1271 | 1038 | 2011 | 1920 | 0.8339 | 129 | 625 | Rust audio pipeline ready case 43 |  |
| 44 | 1272 | 1043 | 2014 | 1920 | 1.0000 | 129 | 625 | Apple Silicon speech check case 44 |  |
| 45 | 1258 | 1045 | 2011 | 1920 | 1.0000 | 129 | 625 | Streaming adapter test case 45 |  |
| 46 | 1259 | 1037 | 2011 | 1920 | 1.0000 | 129 | 625 | Whisper transcription pass case 46 |  |
| 47 | 1258 | 1039 | 2007 | 1920 | 1.0000 | 129 | 625 | Qwen speech synthesis pass case 47 |  |
| 48 | 1261 | 1039 | 2012 | 1920 | 1.0000 | 129 | 625 | Benchmark sample complete case 48 |  |
| 49 | 1271 | 1040 | 2100 | 1920 | 0.9149 | 129 | 625 | Realtime voice path active case 49 |  |
| 50 | 1474 | 1024 | 2056 | 1920 | 1.0000 | 129 | 625 | Vona local inference check case 50 |  |
| 51 | 1261 | 1046 | 2026 | 1920 | 0.9737 | 129 | 625 | Hello from native Vona MLX case 51 |  |
| 52 | 1260 | 1021 | 2019 | 1920 | 1.0000 | 129 | 625 | Local voice test for Ollama case 52 |  |
| 53 | 1241 | 1041 | 2021 | 1920 | 1.0000 | 129 | 625 | Rust audio pipeline ready case 53 |  |
| 54 | 1270 | 1040 | 2019 | 1920 | 1.0000 | 129 | 625 | Apple Silicon speech check case 54 |  |
| 55 | 1256 | 1040 | 2016 | 1920 | 0.9649 | 129 | 625 | Streaming adapter test case 55 |  |
| 56 | 1259 | 1038 | 2015 | 1920 | 1.0000 | 129 | 625 | Whisper transcription pass case 56 |  |
| 57 | 1254 | 1037 | 2023 | 1920 | 0.9521 | 129 | 625 | Qwen speech synthesis pass case 57 |  |
| 58 | 1255 | 1046 | 2017 | 1920 | 1.0000 | 129 | 625 | Benchmark sample complete case 58 |  |
| 59 | 1251 | 1034 | 2020 | 1920 | 1.0000 | 129 | 625 | Realtime voice path active case 59 |  |
| 60 | 1262 | 1043 | 2148 | 1920 | 1.0000 | 129 | 625 | Vona local inference check case 60 |  |
| 61 | 1258 | 1037 | 2038 | 1920 | 1.0000 | 129 | 625 | Hello from native Vona MLX case 61 |  |
| 62 | 1265 | 1040 | 2029 | 1920 | 1.0000 | 129 | 625 | Local voice test for Ollama case 62 |  |
| 63 | 1269 | 1039 | 2093 | 1920 | 1.0000 | 129 | 625 | Rust audio pipeline ready case 63 |  |
| 64 | 1268 | 1040 | 2042 | 1920 | 1.0000 | 129 | 625 | Apple Silicon speech check case 64 |  |
| 65 | 1269 | 1035 | 2030 | 1920 | 1.0000 | 129 | 625 | Streaming adapter test case 65 |  |
| 66 | 1270 | 1037 | 2040 | 1920 | 1.0000 | 129 | 625 | Whisper transcription pass case 66 |  |
| 67 | 1257 | 1036 | 2060 | 1920 | 1.0000 | 129 | 625 | Qwen speech synthesis pass case 67 |  |
| 68 | 1249 | 1036 | 2056 | 1920 | 1.0000 | 129 | 625 | Benchmark sample complete case 68 |  |
| 69 | 1265 | 1035 | 2064 | 1920 | 0.8907 | 129 | 625 | Realtime voice path active case 69 |  |
| 70 | 1267 | 1040 | 2067 | 1920 | 1.0000 | 129 | 625 | Vona local inference check case 70 |  |
| 71 | 1265 | 1040 | 2092 | 1920 | 1.0000 | 129 | 625 | Hello from native Vona MLX case 71 |  |
| 72 | 1258 | 1038 | 2456 | 1920 | 1.0000 | 129 | 625 | Local voice test for Ollama case 72 |  |
| 73 | 1197 | 1037 | 2084 | 1920 | 1.0000 | 129 | 625 | Rust audio pipeline ready case 73 |  |
| 74 | 1267 | 1033 | 2136 | 1920 | 1.0000 | 129 | 625 | Apple Silicon speech check case 74 |  |
| 75 | 1255 | 1040 | 2122 | 1920 | 1.0000 | 129 | 625 | Streaming adapter test case 75 |  |
| 76 | 1260 | 1040 | 2134 | 1920 | 1.0000 | 129 | 625 | Whisper transcription pass case 76 |  |
| 77 | 1267 | 1046 | 2125 | 1920 | 1.0000 | 129 | 625 | Qwen speech synthesis pass case 77 |  |
| 78 | 1271 | 1035 | 2136 | 1920 | 1.0000 | 129 | 625 | Benchmark sample complete case 78 |  |
| 79 | 1274 | 1042 | 2168 | 1920 | 1.0000 | 129 | 625 | Realtime voice path active case 79 |  |
| 80 | 1265 | 1049 | 2125 | 1920 | 1.0000 | 129 | 625 | Vona local inference check case 80 |  |
| 81 | 1265 | 1039 | 2139 | 1920 | 1.0000 | 129 | 625 | Hello from native Vona MLX case 81 |  |
| 82 | 1270 | 1026 | 2165 | 1920 | 1.0000 | 129 | 625 | Local voice test for Ollama case 82 |  |
| 83 | 1275 | 1035 | 2132 | 1920 | 1.0000 | 129 | 625 | Rust audio pipeline ready case 83 |  |
| 84 | 1264 | 1036 | 2145 | 1920 | 1.0000 | 129 | 625 | Apple Silicon speech check case 84 |  |
| 85 | 1283 | 1039 | 2140 | 1920 | 1.0000 | 129 | 625 | Streaming adapter test case 85 |  |
| 86 | 1278 | 1028 | 2122 | 1920 | 1.0000 | 129 | 625 | Whisper transcription pass case 86 |  |
| 87 | 1282 | 1042 | 2130 | 1920 | 1.0000 | 129 | 625 | Qwen speech synthesis pass case 87 |  |
| 88 | 1267 | 1035 | 2135 | 1920 | 1.0000 | 129 | 625 | Benchmark sample complete case 88 |  |
| 89 | 1244 | 1042 | 2142 | 1920 | 0.7847 | 129 | 625 | Realtime voice path active case 89 |  |
| 90 | 1281 | 1019 | 2180 | 1920 | 1.0000 | 129 | 625 | Vona local inference check case 90 |  |
| 91 | 1241 | 1031 | 2189 | 1920 | 0.6359 | 129 | 625 | Hello from native Vona MLX case 91 |  |
| 92 | 1271 | 1040 | 2123 | 1920 | 1.0000 | 129 | 625 | Local voice test for Ollama case 92 |  |
| 93 | 1270 | 1036 | 2120 | 1920 | 1.0000 | 129 | 625 | Rust audio pipeline ready case 93 |  |
| 94 | 1257 | 1033 | 2126 | 1920 | 1.0000 | 129 | 625 | Apple Silicon speech check case 94 |  |
| 95 | 1232 | 1040 | 2214 | 1920 | 1.0000 | 129 | 625 | Streaming adapter test case 95 |  |
| 96 | 1277 | 1037 | 2207 | 1920 | 1.0000 | 129 | 625 | Whisper transcription pass case 96 |  |
| 97 | 1262 | 1039 | 2146 | 1920 | 0.5815 | 129 | 625 | Qwen speech synthesis pass case 97 |  |
| 98 | 1264 | 1037 | 2165 | 1920 | 0.8144 | 129 | 625 | Benchmark sample complete case 98 |  |
| 99 | 1272 | 1039 | 2205 | 1920 | 0.7176 | 129 | 625 | Realtime voice path active case 99 |  |
| 100 | 1237 | 1025 | 2151 | 1920 | 1.0000 | 129 | 625 | Vona local inference check case 100 |  |
