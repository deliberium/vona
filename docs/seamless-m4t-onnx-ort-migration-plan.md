# Seamless M4T ONNX + ort Migration Plan

## Goal

Remove the Seamless local backend dependency on an embedded Python runtime and run local inference through ONNX Runtime via the ort crate.

## Checklist

- [x] Audit current local backend and Python bridge integration
- [x] Add ONNX runtime dependency surface in vona-seamless
- [x] Implement local ONNX runtime adapter module
- [x] Rewire Seamless local backend to ONNX adapter
- [x] Remove Python bridge dependency from local backend path
- [x] Update production backend documentation and env vars
- [x] Verify with cargo check for vona-seamless and workspace

## Notes

- This migration targets runtime architecture first: native inference path, error mapping, and configuration boundary.
- ONNX graph compatibility depends on exported model artifacts and tensor naming.
