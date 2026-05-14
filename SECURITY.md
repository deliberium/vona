# Security Policy

## Supported Versions

| Version | Supported |
| ------- | --------- |
| 0.1.x   | ✓         |

## Reporting a Vulnerability

**Do not open a public GitHub Issue for security vulnerabilities.**

Please report security issues by emailing **kamba@deliberium.ai** with:

- A description of the vulnerability and its potential impact
- Steps to reproduce or a proof-of-concept (if applicable)
- Any suggested mitigations

You can expect an acknowledgement within 72 hours and a resolution timeline within the response.

## Scope

In scope for responsible disclosure:

- Denial-of-service vectors in the sidecar HTTP/IPC server
- Authentication bypass in `VONA_SIDECAR_AUTH_TOKEN` bearer-token middleware
- Unsafe code in `vona` or adapter crates that can be triggered remotely
- Dependency vulnerabilities with a practical exploitation path in this codebase

Out of scope:

- Vulnerabilities in end-user-controlled ONNX model files (model supply-chain is the operator's responsibility)
- Issues in vendored/patched third-party crates that are already fixed upstream
