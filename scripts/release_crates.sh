#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RELEASE_TYPE="current"
PUBLISH=0
SKIP_GATE=0
ALLOW_DIRTY=0
BOOTSTRAP=0
SKIP_PUBLISHED=0

usage() {
  cat <<'USAGE'
Usage: scripts/release_crates.sh [OPTIONS]

Build, package, and optionally publish Vona crates in dependency order.

Options:
  --release current|patch|minor|major  Version action to apply. Default: current.
  --publish                           Run cargo publish. Default: package dry-run only.
  --skip-gate                         Skip scripts/release_gate.sh.
  --allow-dirty                       Pass --allow-dirty to cargo package/publish.
  --bootstrap                         Allow dry-run packaging to stop at unpublished workspace deps.
  --skip-published                    During publish, skip crate versions already present on crates.io.
  -h, --help                          Show this help.

Environment:
  CARGO_REGISTRY_TOKEN must be set by cargo for real publishing.
USAGE
}

wait_for_crate_available() {
  local crate="$1"
  local version="$2"
  local attempt

  for attempt in $(seq 1 30); do
    if crate_version_exists "$crate" "$version"; then
      echo "${crate} v${version} is visible on crates.io"
      return 0
    fi
    echo "Waiting for ${crate} v${version} to appear on crates.io (${attempt}/30)"
    sleep 10
  done

  echo "Timed out waiting for ${crate} v${version} to appear on crates.io" >&2
  return 1
}

crate_version_exists() {
  local crate="$1"
  local version="$2"

  curl -fsS "https://crates.io/api/v1/crates/${crate}/${version}" >/dev/null 2>&1
}

publish_crate() {
  local crate="$1"
  local attempt
  local log_file

  if [[ "$SKIP_PUBLISHED" -eq 1 ]] && crate_version_exists "$crate" "$VERSION"; then
    echo "${crate} v${VERSION} is already published; skipping."
    wait_for_crate_available "$crate" "$VERSION"
    return 0
  fi

  log_file="$(mktemp)"
  for attempt in $(seq 1 18); do
    echo "Publishing ${crate} (attempt ${attempt}/18)"
    if cargo publish -p "$crate" "${COMMON_ARGS[@]}" 2>&1 | tee "$log_file"; then
      wait_for_crate_available "$crate" "$VERSION"
      rm -f "$log_file"
      return 0
    fi

    if [[ "$SKIP_PUBLISHED" -eq 1 ]] && grep -Eiq "already uploaded|already exists|is already published" "$log_file"; then
      echo "${crate} v${VERSION} was already published; continuing."
      wait_for_crate_available "$crate" "$VERSION"
      rm -f "$log_file"
      return 0
    fi

    if grep -Eiq "no matching package named|failed to select a version|required by package" "$log_file"; then
      echo "Dependency index is not ready for ${crate}; retrying after a short wait." >&2
      sleep 20
      continue
    fi

    rm -f "$log_file"
    return 1
  done

  echo "Timed out publishing ${crate} after dependency-index retries." >&2
  rm -f "$log_file"
  return 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --release)
      RELEASE_TYPE="${2:-}"
      shift 2
      ;;
    --publish)
      PUBLISH=1
      shift
      ;;
    --skip-gate)
      SKIP_GATE=1
      shift
      ;;
    --allow-dirty)
      ALLOW_DIRTY=1
      shift
      ;;
    --bootstrap)
      BOOTSTRAP=1
      shift
      ;;
    --skip-published)
      SKIP_PUBLISHED=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$RELEASE_TYPE" in
  current|patch|minor|major) ;;
  *)
    echo "--release must be one of: current, patch, minor, major" >&2
    exit 2
    ;;
esac

CRATES=()
while IFS= read -r crate; do
  CRATES+=("$crate")
done < <(python3 - <<'PY'
import re
from pathlib import Path

root = Path.cwd()
workspace = (root / "Cargo.toml").read_text()
members_block = re.search(r"members\s*=\s*\[(.*?)\]", workspace, re.S).group(1)
members = re.findall(r'"([^"]+)"', members_block)

publish_order = [
    "vona-core",
    "vona-model-provisioning",
    "vona-openai-realtime",
    "vona-gemini-live",
    "vona-azure-speech",
    "vona-elevenlabs",
    "vona-deepgram",
    "vona-seamless",
    "vona-moshi",
    "vona-test-harness",
    "vona-transport-local",
    "vona-sidecar",
    "vona",
]

names = {}
for member in members:
    manifest = root / member / "Cargo.toml"
    text = manifest.read_text()
    name = re.search(r'^name\s*=\s*"([^"]+)"', text, re.M).group(1)
    names[name] = member

missing = [name for name in publish_order if name not in names]
if missing:
    raise SystemExit(f"release order references missing crates: {missing}")

for name in publish_order:
    print(name)
PY
)

VERSION="$(python3 - "$RELEASE_TYPE" <<'PY'
import re
import sys
from datetime import date
from pathlib import Path

release_type = sys.argv[1]
root = Path.cwd()
workspace_path = root / "Cargo.toml"
workspace = workspace_path.read_text()
current = re.search(r'^version\s*=\s*"(\d+)\.(\d+)\.(\d+)"', workspace, re.M)
if not current:
    raise SystemExit("could not find workspace.package version")

major, minor, patch = map(int, current.groups())
if release_type == "major":
    major += 1
    minor = 0
    patch = 0
elif release_type == "minor":
    minor += 1
    patch = 0
elif release_type == "patch":
    patch += 1

version = f"{major}.{minor}.{patch}"
old_version = current.group(0).split('"')[1]

if version != old_version:
    workspace = re.sub(
        r'^version\s*=\s*"\d+\.\d+\.\d+"',
        f'version = "{version}"',
        workspace,
        count=1,
        flags=re.M,
    )
    workspace = re.sub(
        r'(vona[-a-z]*\s*=\s*\{\s*version\s*=\s*)"\d+\.\d+\.\d+"',
        rf'\1"{version}"',
        workspace,
    )
    workspace_path.write_text(workspace)

changelog_path = root / "CHANGELOG.md"
if changelog_path.exists():
    changelog = changelog_path.read_text()
    today = date.today().isoformat()
    heading = f"## [{version}] - {today}"
    existing_version_heading = re.search(rf"^## \[{re.escape(version)}\] - ", changelog, re.M)
    if not existing_version_heading:
        changelog = changelog.replace("## [Unreleased]\n", f"## [Unreleased]\n\n{heading}\n", 1)
    changelog = re.sub(
        r'\[Unreleased\]: https://github\.com/deliberium/vona/compare/v[^.]+\.[^.]+\.[^.\n]+?\.\.\.HEAD',
        f"[Unreleased]: https://github.com/deliberium/vona/compare/v{version}...HEAD",
        changelog,
    )
    if f"[{version}]: https://github.com/deliberium/vona/releases/tag/v{version}" not in changelog:
        changelog += f"\n[{version}]: https://github.com/deliberium/vona/releases/tag/v{version}\n"
    changelog_path.write_text(changelog)

(root / "target").mkdir(exist_ok=True)
(root / "target" / "vona-release-version.txt").write_text(version + "\n")
print(version)
PY
)"

echo "Preparing Vona release v${VERSION} (${RELEASE_TYPE})"

if [[ "$RELEASE_TYPE" != "current" ]]; then
  cargo update --workspace
fi

if [[ "$SKIP_GATE" -eq 0 ]]; then
  bash scripts/release_gate.sh
fi

COMMON_ARGS=()
if [[ "$ALLOW_DIRTY" -eq 1 ]]; then
  COMMON_ARGS+=(--allow-dirty)
fi

if [[ "$PUBLISH" -eq 1 ]]; then
  for crate in "${CRATES[@]}"; do
    publish_crate "$crate"
  done
else
  for crate in "${CRATES[@]}"; do
    echo "Packaging ${crate}"
    if ! cargo package -p "$crate" --offline "${COMMON_ARGS[@]}"; then
      if [[ "$BOOTSTRAP" -eq 1 ]]; then
        echo "Skipping remaining package dry-run checks because ${crate} depends on crates that are not published yet." >&2
        echo "The release gate has already built and tested the workspace locally." >&2
        break
      fi
      exit 1
    fi
  done
  echo "Dry run complete. Re-run with --publish to push crates to crates.io."
fi
