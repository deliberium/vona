//! Moshi binary wire-protocol constants.
//!
//! Reference: <https://github.com/kyutai-labs/moshi/blob/main/rust/protocol.md>
//!
//! Every WebSocket message is a binary frame whose **first byte** is the
//! message type (`MT`).  Remaining bytes are the payload.

/// Message type identifiers (first byte of every binary frame).
pub mod mt {
    /// Handshake: 4-byte protocol version (u32-LE) + 4-byte model version (u32-LE).
    pub const HANDSHAKE: u8 = 0;
    /// Audio: OGG/Opus encoded audio bytes (24 kHz, mono).
    pub const AUDIO: u8 = 1;
    /// Text: UTF-8 string (inner monologue / transcript).
    pub const TEXT: u8 = 2;
    /// Control: single control-byte payload.
    pub const CONTROL: u8 = 3;
    /// Error: UTF-8 error description from the server.
    pub const ERROR: u8 = 5;
}

/// Control byte values (payload of `MT=3` messages).
pub mod ctrl {
    /// Signal that the current speaking turn has ended.
    pub const END_TURN: u8 = 1;
    /// Pause the model output.
    pub const PAUSE: u8 = 2;
    /// Restart / resume the model.
    pub const RESTART: u8 = 3;
}

/// OGG serial number used for all packets in a session (arbitrary constant).
pub const OGG_SERIAL: u32 = 42;

/// Moshi's native sample rate (Hz).
pub const MOSHI_SAMPLE_RATE: u32 = 24_000;

/// Opus frame size in samples (40 ms × 24 000 Hz = 960).
pub const OPUS_FRAME_SAMPLES: usize = 960;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_type_constants_are_distinct() {
        let all = [mt::HANDSHAKE, mt::AUDIO, mt::TEXT, mt::CONTROL, mt::ERROR];
        let unique: std::collections::HashSet<u8> = all.iter().copied().collect();
        assert_eq!(unique.len(), all.len(), "MT constants must be unique");
    }

    #[test]
    fn control_constants_are_distinct() {
        let all = [ctrl::END_TURN, ctrl::PAUSE, ctrl::RESTART];
        let unique: std::collections::HashSet<u8> = all.iter().copied().collect();
        assert_eq!(unique.len(), all.len(), "ctrl constants must be unique");
    }

    #[test]
    fn moshi_sample_rate_is_24k() {
        assert_eq!(MOSHI_SAMPLE_RATE, 24_000);
    }

    #[test]
    fn opus_frame_samples_is_40ms_at_24k() {
        // 40 ms × 24 000 Hz = 960 samples.
        assert_eq!(
            OPUS_FRAME_SAMPLES,
            (MOSHI_SAMPLE_RATE as usize) * 40 / 1_000
        );
    }
}
