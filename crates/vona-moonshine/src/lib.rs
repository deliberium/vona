use std::{
    ffi::{CStr, CString, c_char, c_float, c_int, c_uint, c_ulonglong},
    path::PathBuf,
    process::Stdio,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use async_trait::async_trait;
use libloading::Library;
use serde::{Deserialize, Serialize};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines},
    process::{Child, ChildStdin, ChildStdout, Command},
    sync::Mutex as AsyncMutex,
};
use vona_core::{AudioInputFrame, AudioProcessingError, AudioTranscriber};

pub const DEFAULT_MOONSHINE_SAMPLE_RATE_HZ: u32 = 16_000;
pub const DEFAULT_MOONSHINE_ARCH: &str = "MEDIUM_STREAMING";
const MOONSHINE_HEADER_VERSION: c_int = 20_000;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TranscriptHotword {
    pub replacement: String,
    pub variants: Vec<String>,
}

impl TranscriptHotword {
    pub fn new(
        replacement: impl Into<String>,
        variants: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self {
            replacement: replacement.into(),
            variants: variants.into_iter().map(Into::into).collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MoonshineTranscriberConfig {
    pub python: String,
    pub worker_script: PathBuf,
    pub native_library_path: Option<PathBuf>,
    pub model_path: Option<PathBuf>,
    pub model_arch: String,
    pub cache_root: Option<PathBuf>,
    pub hotwords: Vec<TranscriptHotword>,
}

impl Default for MoonshineTranscriberConfig {
    fn default() -> Self {
        Self {
            python: std::env::var("VONA_MOONSHINE_PYTHON")
                .unwrap_or_else(|_| "python3".to_string()),
            worker_script: std::env::var_os("VONA_MOONSHINE_WORKER_SCRIPT")
                .map(PathBuf::from)
                .unwrap_or_else(default_worker_script),
            native_library_path: std::env::var_os("VONA_MOONSHINE_LIBRARY_PATH").map(PathBuf::from),
            model_path: std::env::var_os("VONA_MOONSHINE_MODEL_PATH").map(PathBuf::from),
            model_arch: std::env::var("VONA_MOONSHINE_ARCH")
                .unwrap_or_else(|_| DEFAULT_MOONSHINE_ARCH.to_string()),
            cache_root: std::env::var_os("VONA_MOONSHINE_CACHE_ROOT").map(PathBuf::from),
            hotwords: default_transcript_hotwords(),
        }
    }
}

#[derive(Clone)]
pub struct NativeMoonshineTranscriber {
    inner: Arc<std::sync::Mutex<NativeMoonshineInner>>,
    hotwords: Arc<Vec<TranscriptHotword>>,
}

impl NativeMoonshineTranscriber {
    pub fn load(config: MoonshineTranscriberConfig) -> Result<Self, AudioProcessingError> {
        let inner = NativeMoonshineInner::load(&config)?;
        Ok(Self {
            inner: Arc::new(std::sync::Mutex::new(inner)),
            hotwords: Arc::new(config.hotwords),
        })
    }

    pub fn from_env() -> Result<Self, AudioProcessingError> {
        Self::load(MoonshineTranscriberConfig::from_env())
    }

    pub fn transcribe_samples_blocking(
        &self,
        samples: &[f32],
        sample_rate_hz: u32,
        channels: u16,
    ) -> Result<String, AudioProcessingError> {
        if sample_rate_hz != DEFAULT_MOONSHINE_SAMPLE_RATE_HZ {
            return Err(AudioProcessingError::InvalidInput(format!(
                "native Moonshine expects {DEFAULT_MOONSHINE_SAMPLE_RATE_HZ} Hz audio, got {sample_rate_hz} Hz"
            )));
        }
        if channels != 1 {
            return Err(AudioProcessingError::InvalidInput(format!(
                "native Moonshine expects mono audio, got {channels} channels"
            )));
        }
        let mut inner = self.inner.lock().map_err(|_| {
            AudioProcessingError::Runtime("native Moonshine transcriber lock poisoned".to_string())
        })?;
        let transcript = inner.transcribe(samples, sample_rate_hz)?;
        Ok(postprocess_transcript(&transcript, &self.hotwords))
    }
}

#[async_trait]
impl AudioTranscriber for NativeMoonshineTranscriber {
    async fn transcribe_audio(
        &self,
        input: AudioInputFrame,
    ) -> Result<String, AudioProcessingError> {
        let this = self.clone();
        tokio::task::spawn_blocking(move || {
            this.transcribe_samples_blocking(&input.samples, input.sample_rate_hz, input.channels)
        })
        .await
        .map_err(|error| {
            AudioProcessingError::Runtime(format!("native Moonshine task join failed: {error}"))
        })?
    }
}

impl MoonshineTranscriberConfig {
    pub fn from_env() -> Self {
        let mut config = Self::default();
        if let Some(hotwords) = transcript_hotwords_from_env("VONA_MOONSHINE_HOTWORDS") {
            config.hotwords = hotwords;
        }
        config
    }
}

#[derive(Clone)]
pub struct ProtectedMoonshineTranscriber {
    config: MoonshineTranscriberConfig,
    worker: Arc<AsyncMutex<Option<MoonshineWorker>>>,
    next_request_id: Arc<AtomicU64>,
}

impl ProtectedMoonshineTranscriber {
    pub fn new(config: MoonshineTranscriberConfig) -> Self {
        Self {
            config,
            worker: Arc::new(AsyncMutex::new(None)),
            next_request_id: Arc::new(AtomicU64::new(1)),
        }
    }

    pub fn from_env() -> Self {
        Self::new(MoonshineTranscriberConfig::from_env())
    }

    pub fn config(&self) -> &MoonshineTranscriberConfig {
        &self.config
    }

    pub async fn transcribe_samples(
        &self,
        samples: Vec<f32>,
        sample_rate_hz: u32,
        channels: u16,
    ) -> Result<String, AudioProcessingError> {
        if sample_rate_hz != DEFAULT_MOONSHINE_SAMPLE_RATE_HZ {
            return Err(AudioProcessingError::InvalidInput(format!(
                "Moonshine expects {DEFAULT_MOONSHINE_SAMPLE_RATE_HZ} Hz audio, got {sample_rate_hz} Hz"
            )));
        }
        if channels != 1 {
            return Err(AudioProcessingError::InvalidInput(format!(
                "Moonshine expects mono audio, got {channels} channels"
            )));
        }

        let request_id = self.next_request_id.fetch_add(1, Ordering::Relaxed);
        let mut guard = self.worker.lock().await;
        if guard.is_none() {
            *guard = Some(start_moonshine_worker(&self.config).await?);
        }

        let Some(worker) = guard.as_mut() else {
            return Err(AudioProcessingError::Runtime(
                "Moonshine worker was not available after start".to_string(),
            ));
        };

        match send_moonshine_worker_request(worker, request_id, sample_rate_hz, channels, &samples)
            .await
        {
            Ok(transcript) => Ok(postprocess_transcript(&transcript, &self.config.hotwords)),
            Err(error) => {
                *guard = None;
                Err(error)
            }
        }
    }
}

#[async_trait]
impl AudioTranscriber for ProtectedMoonshineTranscriber {
    async fn transcribe_audio(
        &self,
        input: AudioInputFrame,
    ) -> Result<String, AudioProcessingError> {
        self.transcribe_samples(input.samples, input.sample_rate_hz, input.channels)
            .await
    }
}

struct NativeMoonshineInner {
    api: NativeMoonshineApi,
    handle: c_int,
}

impl NativeMoonshineInner {
    fn load(config: &MoonshineTranscriberConfig) -> Result<Self, AudioProcessingError> {
        let library_path = config.native_library_path.clone().ok_or_else(|| {
            AudioProcessingError::ModelUnavailable(
                "VONA_MOONSHINE_LIBRARY_PATH is required for native Moonshine".to_string(),
            )
        })?;
        let model_path = config.model_path.as_ref().ok_or_else(|| {
            AudioProcessingError::ModelUnavailable(
                "VONA_MOONSHINE_MODEL_PATH is required for native Moonshine".to_string(),
            )
        })?;
        let api = NativeMoonshineApi::load(library_path)?;
        let model_path = CString::new(model_path.to_string_lossy().as_bytes()).map_err(|_| {
            AudioProcessingError::InvalidInput("Moonshine model path contains NUL byte".to_string())
        })?;
        let model_arch = model_arch_value(&config.model_arch)?;
        let handle = unsafe {
            (api.moonshine_load_transcriber_from_files)(
                model_path.as_ptr(),
                model_arch,
                std::ptr::null(),
                0,
                MOONSHINE_HEADER_VERSION,
            )
        };
        if handle < 0 {
            return Err(api.error(handle, "native Moonshine transcriber load failed"));
        }
        Ok(Self { api, handle })
    }

    fn transcribe(
        &mut self,
        samples: &[f32],
        sample_rate_hz: u32,
    ) -> Result<String, AudioProcessingError> {
        let mut transcript: *mut TranscriptC = std::ptr::null_mut();
        let error = unsafe {
            (self.api.moonshine_transcribe_without_streaming)(
                self.handle,
                samples.as_ptr(),
                samples.len() as c_ulonglong,
                sample_rate_hz as c_int,
                0,
                &mut transcript,
            )
        };
        if error < 0 {
            return Err(self
                .api
                .error(error, "native Moonshine transcription failed"));
        }
        Ok(unsafe { parse_transcript_text(transcript) })
    }
}

impl Drop for NativeMoonshineInner {
    fn drop(&mut self) {
        unsafe {
            (self.api.moonshine_free_transcriber)(self.handle);
        }
    }
}

struct NativeMoonshineApi {
    _library: Library,
    moonshine_error_to_string: unsafe extern "C" fn(c_int) -> *const c_char,
    moonshine_load_transcriber_from_files: unsafe extern "C" fn(
        *const c_char,
        c_uint,
        *const MoonshineOptionC,
        c_ulonglong,
        c_int,
    ) -> c_int,
    moonshine_free_transcriber: unsafe extern "C" fn(c_int),
    moonshine_transcribe_without_streaming: unsafe extern "C" fn(
        c_int,
        *const c_float,
        c_ulonglong,
        c_int,
        c_uint,
        *mut *mut TranscriptC,
    ) -> c_int,
}

unsafe impl Send for NativeMoonshineApi {}

impl NativeMoonshineApi {
    fn load(path: PathBuf) -> Result<Self, AudioProcessingError> {
        let library = unsafe { Library::new(&path) }.map_err(|error| {
            AudioProcessingError::ModelUnavailable(format!(
                "failed to load native Moonshine library {}: {error}",
                path.display()
            ))
        })?;
        unsafe {
            let moonshine_error_to_string = *library
                .get::<unsafe extern "C" fn(c_int) -> *const c_char>(b"moonshine_error_to_string\0")
                .map_err(symbol_error("moonshine_error_to_string"))?;
            let moonshine_load_transcriber_from_files = *library
                .get::<unsafe extern "C" fn(
                    *const c_char,
                    c_uint,
                    *const MoonshineOptionC,
                    c_ulonglong,
                    c_int,
                ) -> c_int>(b"moonshine_load_transcriber_from_files\0")
                .map_err(symbol_error("moonshine_load_transcriber_from_files"))?;
            let moonshine_free_transcriber = *library
                .get::<unsafe extern "C" fn(c_int)>(b"moonshine_free_transcriber\0")
                .map_err(symbol_error("moonshine_free_transcriber"))?;
            let moonshine_transcribe_without_streaming = *library
                .get::<unsafe extern "C" fn(
                    c_int,
                    *const c_float,
                    c_ulonglong,
                    c_int,
                    c_uint,
                    *mut *mut TranscriptC,
                ) -> c_int>(b"moonshine_transcribe_without_streaming\0")
                .map_err(symbol_error("moonshine_transcribe_without_streaming"))?;
            Ok(Self {
                _library: library,
                moonshine_error_to_string,
                moonshine_load_transcriber_from_files,
                moonshine_free_transcriber,
                moonshine_transcribe_without_streaming,
            })
        }
    }

    fn error(&self, code: c_int, context: &'static str) -> AudioProcessingError {
        let message = unsafe {
            let ptr = (self.moonshine_error_to_string)(code);
            if ptr.is_null() {
                "unknown Moonshine error".to_string()
            } else {
                CStr::from_ptr(ptr).to_string_lossy().into_owned()
            }
        };
        AudioProcessingError::Inference(format!("{context}: {message} ({code})"))
    }
}

fn symbol_error(name: &'static str) -> impl FnOnce(libloading::Error) -> AudioProcessingError {
    move |error| {
        AudioProcessingError::ModelUnavailable(format!(
            "native Moonshine library missing symbol {name}: {error}"
        ))
    }
}

#[repr(C)]
struct MoonshineOptionC {
    name: *const c_char,
    value: *const c_char,
}

#[repr(C)]
struct TranscriptWordC {
    text: *const c_char,
    start: c_float,
    end: c_float,
    confidence: c_float,
}

#[repr(C)]
struct TranscriptLineC {
    text: *const c_char,
    audio_data: *const c_float,
    audio_data_count: usize,
    start_time: c_float,
    duration: c_float,
    id: u64,
    is_complete: i8,
    is_updated: i8,
    is_new: i8,
    has_text_changed: i8,
    has_speaker_id: i8,
    speaker_id: u64,
    speaker_index: u32,
    last_transcription_latency_ms: u32,
    words: *const TranscriptWordC,
    word_count: u64,
}

#[repr(C)]
struct TranscriptC {
    lines: *const TranscriptLineC,
    line_count: u64,
}

unsafe fn parse_transcript_text(transcript: *const TranscriptC) -> String {
    if transcript.is_null() {
        return String::new();
    }
    let transcript = unsafe { &*transcript };
    if transcript.lines.is_null() || transcript.line_count == 0 {
        return String::new();
    }
    let lines =
        unsafe { std::slice::from_raw_parts(transcript.lines, transcript.line_count as usize) };
    lines
        .iter()
        .filter_map(|line| {
            if line.text.is_null() {
                None
            } else {
                Some(
                    unsafe { CStr::from_ptr(line.text) }
                        .to_string_lossy()
                        .trim()
                        .to_string(),
                )
            }
        })
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

fn model_arch_value(value: &str) -> Result<c_uint, AudioProcessingError> {
    match value.trim().to_ascii_uppercase().as_str() {
        "TINY" | "TINY_EN" => Ok(0),
        "BASE" | "BASE_EN" => Ok(1),
        "TINY_STREAMING" | "TINY-STREAMING" => Ok(2),
        "BASE_STREAMING" | "BASE-STREAMING" => Ok(3),
        "SMALL_STREAMING" | "SMALL-STREAMING" => Ok(4),
        "MEDIUM_STREAMING" | "MEDIUM-STREAMING" => Ok(5),
        other => Err(AudioProcessingError::InvalidInput(format!(
            "unsupported Moonshine arch {other:?}"
        ))),
    }
}

struct MoonshineWorker {
    _child: Child,
    stdin: ChildStdin,
    stdout: Lines<BufReader<ChildStdout>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct MoonshineWorkerReady {
    ready: bool,
    arch: String,
    model_path: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct MoonshineWorkerRequest {
    id: u64,
    sample_rate_hz: u32,
    channels: u16,
    samples: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct MoonshineWorkerResponse {
    id: u64,
    transcript: Option<String>,
    error: Option<String>,
}

async fn start_moonshine_worker(
    config: &MoonshineTranscriberConfig,
) -> Result<MoonshineWorker, AudioProcessingError> {
    let mut command = Command::new(&config.python);
    command
        .arg(&config.worker_script)
        .arg("--arch")
        .arg(&config.model_arch)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());
    if let Some(cache_root) = &config.cache_root {
        command.arg("--cache-root").arg(cache_root);
    }
    command.kill_on_drop(true);

    let mut child = command.spawn().map_err(|error| {
        AudioProcessingError::Runtime(format!(
            "failed to spawn Moonshine worker {} {}: {error}",
            config.python,
            config.worker_script.display()
        ))
    })?;
    let stdin = child.stdin.take().ok_or_else(|| {
        AudioProcessingError::Runtime("Moonshine worker stdin missing".to_string())
    })?;
    let stdout = child.stdout.take().ok_or_else(|| {
        AudioProcessingError::Runtime("Moonshine worker stdout missing".to_string())
    })?;
    let mut worker = MoonshineWorker {
        _child: child,
        stdin,
        stdout: BufReader::new(stdout).lines(),
    };
    let ready = worker
        .stdout
        .next_line()
        .await
        .map_err(|error| {
            AudioProcessingError::Runtime(format!("Moonshine worker ready read failed: {error}"))
        })?
        .ok_or_else(|| {
            AudioProcessingError::Runtime("Moonshine worker exited before ready".to_string())
        })?;
    let ready: MoonshineWorkerReady = serde_json::from_str(&ready).map_err(|error| {
        AudioProcessingError::Runtime(format!("Moonshine worker ready JSON invalid: {error}"))
    })?;
    if !ready.ready {
        return Err(AudioProcessingError::Runtime(format!(
            "Moonshine worker did not report ready for arch {}",
            ready.arch
        )));
    }
    Ok(worker)
}

async fn send_moonshine_worker_request(
    worker: &mut MoonshineWorker,
    request_id: u64,
    sample_rate_hz: u32,
    channels: u16,
    samples: &[f32],
) -> Result<String, AudioProcessingError> {
    let header = serde_json::to_string(&MoonshineWorkerRequest {
        id: request_id,
        sample_rate_hz,
        channels,
        samples: samples.len(),
    })
    .map_err(|error| {
        AudioProcessingError::Runtime(format!("Moonshine request JSON failed: {error}"))
    })?;
    worker
        .stdin
        .write_all(header.as_bytes())
        .await
        .map_err(|error| {
            AudioProcessingError::Runtime(format!("Moonshine worker header write failed: {error}"))
        })?;
    worker.stdin.write_all(b"\n").await.map_err(|error| {
        AudioProcessingError::Runtime(format!("Moonshine worker header write failed: {error}"))
    })?;
    write_f32_le(&mut worker.stdin, samples).await?;
    worker.stdin.flush().await.map_err(|error| {
        AudioProcessingError::Runtime(format!("Moonshine worker flush failed: {error}"))
    })?;

    let response = worker
        .stdout
        .next_line()
        .await
        .map_err(|error| {
            AudioProcessingError::Runtime(format!("Moonshine worker response read failed: {error}"))
        })?
        .ok_or_else(|| {
            AudioProcessingError::Runtime(
                "Moonshine worker exited during transcription".to_string(),
            )
        })?;
    let response: MoonshineWorkerResponse = serde_json::from_str(&response).map_err(|error| {
        AudioProcessingError::Runtime(format!("Moonshine worker response JSON invalid: {error}"))
    })?;
    if response.id != request_id {
        return Err(AudioProcessingError::Runtime(format!(
            "Moonshine worker response id {} did not match request id {request_id}",
            response.id
        )));
    }
    if let Some(error) = response.error {
        return Err(AudioProcessingError::Inference(error));
    }
    response.transcript.ok_or_else(|| {
        AudioProcessingError::Runtime("Moonshine worker response omitted transcript".to_string())
    })
}

async fn write_f32_le(stdin: &mut ChildStdin, samples: &[f32]) -> Result<(), AudioProcessingError> {
    let mut bytes = Vec::with_capacity(samples.len() * 4);
    for sample in samples {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    stdin.write_all(&bytes).await.map_err(|error| {
        AudioProcessingError::Runtime(format!("Moonshine worker PCM write failed: {error}"))
    })
}

fn default_worker_script() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scripts/moonshine_worker.py")
}

pub fn default_transcript_hotwords() -> Vec<TranscriptHotword> {
    vec![
        TranscriptHotword::new("Vona", ["vona", "voner", "vowna"]),
        TranscriptHotword::new("Moonshine", ["moon shine"]),
        TranscriptHotword::new("Deepgram", ["deep gram"]),
        TranscriptHotword::new("Lumina", ["luminous"]),
        TranscriptHotword::new("CloudPool", ["cloud pool", "cloud-pool"]),
        TranscriptHotword::new("backend", ["back end"]),
        TranscriptHotword::new("Ollama", ["ollama", "alama", "allama"]),
        TranscriptHotword::new("Qwen", ["qwen", "qn", "q-n"]),
    ]
}

pub fn transcript_hotwords_from_env(name: &str) -> Option<Vec<TranscriptHotword>> {
    std::env::var(name)
        .ok()
        .and_then(|value| parse_transcript_hotwords(&value).ok())
}

pub fn parse_transcript_hotwords(value: &str) -> Result<Vec<TranscriptHotword>, String> {
    let mut hotwords = Vec::new();
    for entry in value.split(',') {
        let entry = entry.trim();
        if entry.is_empty() {
            continue;
        }
        let Some((replacement, variants)) = entry.split_once('=') else {
            return Err(format!(
                "invalid hotword entry {entry:?}; expected replacement=variant|variant"
            ));
        };
        let replacement = replacement.trim();
        if replacement.is_empty() {
            return Err("hotword replacement cannot be empty".to_string());
        }
        let variants = variants
            .split('|')
            .map(str::trim)
            .filter(|variant| !variant.is_empty())
            .map(str::to_string)
            .collect::<Vec<_>>();
        if variants.is_empty() {
            return Err(format!("hotword {replacement:?} has no variants"));
        }
        hotwords.push(TranscriptHotword::new(replacement, variants));
    }
    Ok(hotwords)
}

pub fn postprocess_transcript(transcript: &str, hotwords: &[TranscriptHotword]) -> String {
    let mut output = transcript.to_string();
    for hotword in hotwords {
        for variant in &hotword.variants {
            output = replace_case_insensitive_wordish(&output, variant, &hotword.replacement);
        }
    }
    output
}

fn replace_case_insensitive_wordish(input: &str, needle: &str, replacement: &str) -> String {
    let lower_input = input.to_ascii_lowercase();
    let lower_needle = needle.to_ascii_lowercase();
    if lower_needle.is_empty() || !lower_input.contains(&lower_needle) {
        return input.to_string();
    }

    let mut output = String::with_capacity(input.len());
    let mut index = 0usize;
    while let Some(offset) = lower_input[index..].find(&lower_needle) {
        let start = index + offset;
        let end = start + lower_needle.len();
        let before = input[..start].chars().next_back();
        let after = input[end..].chars().next();
        if before.is_some_and(is_word_char) || after.is_some_and(is_word_char) {
            output.push_str(&input[index..end]);
        } else {
            output.push_str(&input[index..start]);
            output.push_str(replacement);
        }
        index = end;
    }
    output.push_str(&input[index..]);
    output
}

fn is_word_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || ch == '_'
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_hotwords() {
        let hotwords = parse_transcript_hotwords("Lumina=luminous,Deepgram=deep gram|d gram")
            .expect("hotwords parse");
        assert_eq!(hotwords.len(), 2);
        assert_eq!(hotwords[1].variants, ["deep gram", "d gram"]);
    }

    #[test]
    fn postprocess_replaces_product_terms_without_touching_inside_words() {
        let text = "Inspect the luminous state and the deep gram voice back end.";
        let output = postprocess_transcript(text, &default_transcript_hotwords());
        assert_eq!(
            output,
            "Inspect the Lumina state and the Deepgram voice backend."
        );
        let untouched =
            postprocess_transcript("a backendless example", &default_transcript_hotwords());
        assert_eq!(untouched, "a backendless example");
    }

    #[test]
    fn parses_model_arch_aliases() {
        assert_eq!(model_arch_value("MEDIUM_STREAMING").unwrap(), 5);
        assert_eq!(model_arch_value("small-streaming").unwrap(), 4);
        assert!(model_arch_value("giant").is_err());
    }
}
