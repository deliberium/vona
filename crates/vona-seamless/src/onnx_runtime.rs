use ndarray::{Array2, IxDyn};
use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::TensorRef,
};
use tokio::sync::Mutex;
use vona_core::BackendError;

use crate::local::SeamlessM4tLocalConfig;

pub struct SeamlessM4tOnnxRuntime {
    session: Mutex<Session>,
    input_name: String,
    output_name: String,
    output_sample_rate_hz: u32,
}

impl SeamlessM4tOnnxRuntime {
    pub fn new(config: &SeamlessM4tLocalConfig) -> Result<Self, BackendError> {
        let model_path = config.onnx_model_path.as_ref().ok_or_else(|| {
            BackendError::Start(
                "missing VONA_STS_ONNX_MODEL_PATH; set it to a Seamless M4T ONNX model file"
                    .to_string(),
            )
        })?;

        let session = Session::builder()
            .map_err(|err| BackendError::Start(format!("onnx session builder failed: {err}")))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|err| {
                BackendError::Start(format!("onnx graph optimization setup failed: {err}"))
            })?
            .commit_from_file(model_path)
            .map_err(|err| {
                BackendError::Start(format!("failed to load ONNX model at {model_path}: {err}"))
            })?;

        Ok(Self {
            session: Mutex::new(session),
            input_name: config.onnx_input_name.clone(),
            output_name: config.onnx_output_name.clone(),
            output_sample_rate_hz: config.onnx_sample_rate_hz,
        })
    }

    pub async fn run_audio_step(
        &self,
        input_samples: &[f32],
        input_sample_rate_hz: u32,
    ) -> Result<Vec<f32>, BackendError> {
        let normalized = resample_mono(
            input_samples,
            input_sample_rate_hz,
            self.output_sample_rate_hz,
        );

        let input_tensor = Array2::from_shape_vec((1, normalized.len()), normalized)
            .map_err(|err| BackendError::Step(format!("onnx input tensor shape failed: {err}")))?;
        let input_value = TensorRef::from_array_view(input_tensor.view())
            .map_err(|err| BackendError::Step(format!("onnx tensor view build failed: {err}")))?;

        let mut session = self.session.lock().await;
        let outputs = session
            .run(ort::inputs![&self.input_name => input_value])
            .map_err(|err| BackendError::Step(format!("onnx inference failed: {err}")))?;

        let tensor = outputs
            .get(&self.output_name)
            .ok_or_else(|| {
                BackendError::Step(format!(
                    "onnx output tensor '{}' not found in model response",
                    self.output_name
                ))
            })?
            .try_extract_array::<f32>()
            .map_err(|err| BackendError::Step(format!("onnx output extraction failed: {err}")))?;

        Ok(tensor
            .view()
            .into_dimensionality::<IxDyn>()
            .map_err(|err| {
                BackendError::Step(format!("onnx output dimension conversion failed: {err}"))
            })?
            .iter()
            .copied()
            .collect())
    }
}

pub fn resample_mono(input: &[f32], src_hz: u32, dst_hz: u32) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }
    if src_hz == dst_hz {
        return input.to_vec();
    }

    let ratio = dst_hz as f64 / src_hz as f64;
    let out_len = ((input.len() as f64) * ratio).round().max(1.0) as usize;

    let mut output = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src_pos = (i as f64) / ratio;
        let idx = src_pos.floor() as usize;
        let frac = (src_pos - idx as f64) as f32;

        let a = input.get(idx).copied().unwrap_or(0.0);
        let b = input.get(idx + 1).copied().unwrap_or(a);
        output.push(a + (b - a) * frac);
    }

    output
}

#[cfg(test)]
mod tests {
    use super::resample_mono;

    #[test]
    fn resample_empty_returns_empty() {
        assert!(resample_mono(&[], 16_000, 24_000).is_empty());
    }

    #[test]
    fn resample_identity_rate_returns_clone() {
        let input: Vec<f32> = (0..8).map(|i| i as f32 * 0.1).collect();
        let output = resample_mono(&input, 16_000, 16_000);
        assert_eq!(output, input);
    }

    #[test]
    fn resample_upsample_output_len() {
        // 8 kHz → 16 kHz should double the length (approximately)
        let input: Vec<f32> = vec![0.0f32; 100];
        let output = resample_mono(&input, 8_000, 16_000);
        assert_eq!(output.len(), 200);
    }

    #[test]
    fn resample_downsample_output_len() {
        // 48 kHz → 16 kHz should produce 1/3 the samples (approximately)
        let input: Vec<f32> = vec![0.0f32; 480];
        let output = resample_mono(&input, 48_000, 16_000);
        assert_eq!(output.len(), 160);
    }

    #[test]
    fn resample_dc_signal_preserved() {
        // A constant 1.0 signal should stay 1.0 after any resampling ratio.
        let input = vec![1.0f32; 64];
        let output = resample_mono(&input, 8_000, 24_000);
        for &sample in &output {
            assert!((sample - 1.0).abs() < 1e-5, "expected 1.0 but got {sample}");
        }
    }

    #[test]
    fn resample_single_sample_does_not_panic() {
        let output = resample_mono(&[0.5], 16_000, 48_000);
        assert!(!output.is_empty());
    }
}
