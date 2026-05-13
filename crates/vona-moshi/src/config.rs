/// Configuration for the Moshi WebSocket backend.
///
/// The Moshi inference server (`moshi-backend` from kyutai-labs/moshi) must
/// already be running before creating a session.  Start it with:
///
/// ```text
/// # CUDA:
/// cargo run --features cuda --bin moshi-backend -r -- \
///     --config moshi-backend/config.json standalone
///
/// # Metal (macOS):
/// cargo run --features metal --bin moshi-backend -r -- \
///     --config moshi-backend/config.json standalone
/// ```
///
/// The server listens on `https://127.0.0.1:8998` by default and uses a
/// self-signed certificate, so set `accept_invalid_certs = true` for local
/// development.
#[derive(Debug, Clone)]
pub struct MoshiConfig {
    /// WebSocket endpoint for the Moshi API, e.g. `wss://127.0.0.1:8998/api/chat`.
    pub url: String,

    /// Accept self-signed TLS certificates.  **Never set this to `true` in
    /// production** — only appropriate for connecting to a local dev server.
    pub accept_invalid_certs: bool,
}

impl Default for MoshiConfig {
    fn default() -> Self {
        Self {
            url: "wss://127.0.0.1:8998/api/chat".into(),
            accept_invalid_certs: false,
        }
    }
}

impl MoshiConfig {
    /// Build a config from environment variables.
    ///
    /// | Variable                      | Default                             |
    /// |-------------------------------|-------------------------------------|
    /// | `MOSHI_URL`                   | `wss://127.0.0.1:8998/api/chat`     |
    /// | `MOSHI_ACCEPT_INVALID_CERTS`  | `""` (false)                        |
    pub fn from_env() -> Self {
        Self {
            url: std::env::var("MOSHI_URL")
                .unwrap_or_else(|_| "wss://127.0.0.1:8998/api/chat".into()),
            accept_invalid_certs: std::env::var("MOSHI_ACCEPT_INVALID_CERTS").ok().as_deref()
                == Some("1"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn default_url_is_local_moshi() {
        let cfg = MoshiConfig::default();
        assert_eq!(cfg.url, "wss://127.0.0.1:8998/api/chat");
    }

    #[test]
    fn default_rejects_invalid_certs() {
        let cfg = MoshiConfig::default();
        assert!(!cfg.accept_invalid_certs);
    }

    #[test]
    fn from_env_picks_up_custom_url() {
        let _guard = ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::set_var("MOSHI_URL", "wss://example.com/api/chat");
        }
        let cfg = MoshiConfig::from_env();
        unsafe {
            std::env::remove_var("MOSHI_URL");
        }
        assert_eq!(cfg.url, "wss://example.com/api/chat");
    }

    #[test]
    fn from_env_accept_invalid_certs_off_by_default() {
        let _guard = ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::remove_var("MOSHI_ACCEPT_INVALID_CERTS");
        }
        let cfg = MoshiConfig::from_env();
        assert!(!cfg.accept_invalid_certs);
    }

    #[test]
    fn from_env_accept_invalid_certs_enabled_by_one() {
        let _guard = ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::set_var("MOSHI_ACCEPT_INVALID_CERTS", "1");
        }
        let cfg = MoshiConfig::from_env();
        unsafe {
            std::env::remove_var("MOSHI_ACCEPT_INVALID_CERTS");
        }
        assert!(cfg.accept_invalid_certs);
    }

    #[test]
    fn from_env_accept_invalid_certs_not_enabled_by_true() {
        // Only the literal string "1" enables it — "true" should not.
        let _guard = ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::set_var("MOSHI_ACCEPT_INVALID_CERTS", "true");
        }
        let cfg = MoshiConfig::from_env();
        unsafe {
            std::env::remove_var("MOSHI_ACCEPT_INVALID_CERTS");
        }
        assert!(!cfg.accept_invalid_certs);
    }
}
