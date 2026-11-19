#!/usr/bin/env bash
# Package HawaVoClean native macOS CLAP and VST3 CFBundles from the Rust release dylib.
#
# Usage:
#   daw-plugin/bundle-macos.sh [output-dir]
#
# Optional environment variables:
#   CODESIGN_IDENTITY  Codesigning identity (default: "-" for ad-hoc local signing)
#   CARGO_FEATURES     Extra Cargo features to pass when building (e.g. "onnx")
#   SKIP_BUILD         Set to "1" to reuse an existing target/release/libhawavoclean_daw.dylib

set -euo pipefail

export COPYFILE_DISABLE=1

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "error: daw-plugin/bundle-macos.sh must be run on macOS" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${1:-${SCRIPT_DIR}/target/bundled}"
CODESIGN_IDENTITY="${CODESIGN_IDENTITY:--}"
MANIFEST_PATH="${SCRIPT_DIR}/Cargo.toml"
DYLIB_PATH="${SCRIPT_DIR}/target/release/libhawavoclean_daw.dylib"
PLUGIN_NAME="HawaVoClean"

if [[ "${SKIP_BUILD:-0}" != "1" || ! -f "${DYLIB_PATH}" ]]; then
  BUILD_ARGS=(build --release --manifest-path "${MANIFEST_PATH}")
  if [[ -n "${CARGO_FEATURES:-}" ]]; then
    BUILD_ARGS+=(--features "${CARGO_FEATURES}")
  fi
  cargo "${BUILD_ARGS[@]}"
fi

if [[ ! -f "${DYLIB_PATH}" ]]; then
  echo "error: release dylib not found at ${DYLIB_PATH}" >&2
  exit 1
fi

VERSION="$(awk -F'"' '/^version[[:space:]]*=[[:space:]]*"/ { print $2; exit }' "${MANIFEST_PATH}")"
if [[ -z "${VERSION}" ]]; then
  echo "error: failed to read package version from ${MANIFEST_PATH}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

create_bundle() {
  local format="$1"
  local bundle_id="$2"
  local info_string="$3"
  local bundle_dir="${OUT_DIR}/${PLUGIN_NAME}.${format}"
  local contents_dir="${bundle_dir}/Contents"
  local macos_dir="${contents_dir}/MacOS"
  local bin_path="${macos_dir}/${PLUGIN_NAME}"

  rm -rf "${bundle_dir}"
  mkdir -p "${macos_dir}"

  cp -f "${DYLIB_PATH}" "${bin_path}"
  chmod 0755 "${bin_path}"

  # Rewrite LC_ID_DYLIB so no machine-specific target/release path leaks into the bundle.
  install_name_tool -id "@loader_path/${PLUGIN_NAME}" "${bin_path}"

  printf 'BNDL????' > "${contents_dir}/PkgInfo"

  cat > "${contents_dir}/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key>
  <string>English</string>
  <key>CFBundleExecutable</key>
  <string>${PLUGIN_NAME}</string>
  <key>CFBundleGetInfoString</key>
  <string>${info_string}</string>
  <key>CFBundleIdentifier</key>
  <string>${bundle_id}</string>
  <key>CFBundleInfoDictionaryVersion</key>
  <string>6.0</string>
  <key>CFBundleName</key>
  <string>${PLUGIN_NAME}</string>
  <key>CFBundlePackageType</key>
  <string>BNDL</string>
  <key>CFBundleShortVersionString</key>
  <string>${VERSION}</string>
  <key>CFBundleSignature</key>
  <string>????</string>
  <key>CFBundleVersion</key>
  <string>${VERSION}</string>
  <key>LSMinimumSystemVersion</key>
  <string>14.0</string>
  <key>NSHighResolutionCapable</key>
  <true/>
  <key>NSHumanReadableCopyright</key>
  <string>Copyright © HawaVoClean</string>
</dict>
</plist>
EOF

  plutil -lint "${contents_dir}/Info.plist" >/dev/null

  if command -v xattr >/dev/null 2>&1; then
    xattr -cr "${bundle_dir}"
  fi

  codesign --force --sign "${CODESIGN_IDENTITY}" "${bundle_dir}"
  codesign --verify --strict "${bundle_dir}"
}

create_bundle "clap" "com.hawavoclean.daw.clap" "${PLUGIN_NAME} ${VERSION} CLAP Audio Plug-In"
create_bundle "vst3" "com.hawavoclean.daw.vst3" "${PLUGIN_NAME} ${VERSION} VST3 Audio Plug-In"

echo "Bundled macOS plugins in ${OUT_DIR}:"
echo "  - ${OUT_DIR}/${PLUGIN_NAME}.clap"
echo "  - ${OUT_DIR}/${PLUGIN_NAME}.vst3"
