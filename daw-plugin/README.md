# HawaVoClean Native DAW Plugin (AU / VST3 / CLAP)

Real-time, zero-audio-thread-allocation speech restoration plugin built in Rust (`nih-plug`) with verified ONNX export tooling for HawaVoClean's pure-PyTorch models (`MossFormer2_SE_48K`, `DeepFilterNet3`, and `GAP-URGENet`).

## Architecture

- **`src/lib.rs`**:
  - Thread-safe, sample-accurate smoothed DAW parameters:
    - `strength` (`0..100%`)
    - `ambience_floor_db` (`-60..0 dB`)
    - `delta_listen` (boolean — phase-inverted `dry_aligned - wet` audition of removed noise)
    - `wet_dry` (`0..100%`, delay-compensated)
    - `output_gain_db` (`-24..+24 dB`)
  - Wait-free fixed-capacity SPSC ring buffer (`FixedRingBuffer<N>`) + 50%-overlap `sqrt(Hann)` analysis & synthesis COLA frame engine (`960` samples, `480` hop, `481` FFT bins, `960`-sample reported DAW PDC latency = `20.0 ms @ 48 kHz`, matching DeepFilterNet3 causal streaming geometry).
  - Pluggable `FrameProcessor` trait:
    - **`WienerFallbackProcessor`**: self-contained, zero-allocation frequency-domain Wiener denoiser (mixed-radix `Fft960` + asymmetric noise-floor tracker).
    - **`OnnxFrameProcessor`** (`--features onnx`): ONNX Runtime streaming mask processor (`[1, 1, 481]` magnitude -> `[1, 1, 481]` gain mask) supporting CoreML EP (Apple Silicon) and safe CPU / Wiener fallback.
- **`export_onnx.py`**:
  - Standalone CLI exporter for `mossformer2`, `deepfilternet3`, and `gap-urgenet` with preflight diagnostics, SHA-256 sidecar metadata (`.onnx.meta.json`), and PyTorch-vs-ONNX-Runtime float32 parity verification.

---

## 1. Export & Verify ONNX Models

Check your Python environment first:

```bash
python3 daw-plugin/export_onnx.py --preflight-only
```

Install optional ONNX export & parity dependencies if missing:

```bash
uv pip install "onnx>=1.16.0" "onnxruntime>=1.19.0"
```

Export HawaVoClean's flagship `MossFormer2_SE_48K` model and verify float32 parity (`max_abs_diff <= 1e-4`):

```bash
python3 daw-plugin/export_onnx.py \
  --model mossformer2 \
  --output daw-plugin/models/mossformer2_se_48k.onnx \
  --opset 17 \
  --verify
```

Export `GAP-URGENet` neural dereverberation (`16 kHz` TF-GridNetV3):

```bash
python3 daw-plugin/export_onnx.py \
  --model gap-urgenet \
  --output daw-plugin/models/gap_urgenet.onnx \
  --opset 17 \
  --verify
```

Smoke-test the ONNX export pipeline without downloading model checkpoints:

```bash
python3 daw-plugin/export_onnx.py \
  --model mossformer2 \
  --untrained-smoke \
  --output /tmp/mf2_smoke.onnx \
  --verify
```

---

## 2. Build VST3 & CLAP Plugins on macOS (Apple Silicon / Universal)

### Prerequisites

- Rust toolchain (`rustup`, stable 1.80+)
- Xcode Command Line Tools (`xcode-select --install`)
- Optional for ONNX Runtime + CoreML EP: `libonnxruntime.dylib` (`brew install onnxruntime`)

### Compile & Run DSP Unit Tests

```bash
cd daw-plugin
cargo test --release
```

### Build `.vst3` and `.clap` Bundles

Use `nih_plug_xtask` (or bundle directly from `target/release/libhawavoclean_daw.dylib`):

```bash
# Build release dynamic library (fallback Wiener DSP, zero external runtime dependencies)
cargo build --release

# Or build with ONNX Runtime + CoreML Execution Provider support:
cargo build --release --features onnx
```

Package into signed macOS CFBundles (`Info.plist`, `PkgInfo`, rewritten `@loader_path` install name, and ad-hoc `codesign --force --sign -`):

```bash
# Builds release dylib and writes HawaVoClean.clap + HawaVoClean.vst3 into target/bundled/
./bundle-macos.sh

# Or target a custom output directory:
./bundle-macos.sh /tmp/hawa-daw-bundles
```

Install for local DAW testing (Logic Pro via AU wrapper, DaVinci Fairlight, Reaper, Ableton Live, Bitwig):

```bash
cp -R target/bundled/HawaVoClean.vst3 ~/Library/Audio/Plug-Ins/VST3/
cp -R target/bundled/HawaVoClean.clap ~/Library/Audio/Plug-Ins/CLAP/
```

---

## 3. Building an Audio Unit (AUv2) for Logic Pro / GarageBand / Final Cut Pro

`nih-plug` natively exports **VST3** and **CLAP**. To ship a native macOS **Audio Unit (`.component`)** from the exact same CLAP binary:

1. Wrap `HawaVoClean.clap` using the official MIT-licensed [`clap-wrapper`](https://github.com/free-audio/clap-wrapper) (`AUv2` target):
   ```bash
   cmake -B build-au -S /path/to/clap-wrapper \
     -DCLAP_WRAPPER_BUILD_AUV2=ON \
     -DCLAP_WRAPPER_OUTPUT_NAME="HawaVoClean" \
     -DCLAP_WRAPPER_BUNDLE_ID="com.hawavoclean.daw.au" \
     -DCLAP_WRAPPER_CLAP_PATH="$(pwd)/target/bundled/HawaVoClean.clap"
   cmake --build build-au --config Release
   cp -R build-au/HawaVoClean.component ~/Library/Audio/Plug-Ins/Components/
   ```
2. Validate with Apple's Audio Unit validator:
   ```bash
   auval -v aufx HwVc HwVc
   ```

---

## 4. CoreML / ONNX Runtime Execution Provider Notes (macOS)

- Export models with `--opset 17` for broad Apple Neural Engine (ANE) / CoreML EP operator coverage.
- Set `ORT_DYLIB_PATH=/opt/homebrew/lib/libonnxruntime.dylib` (or bundle `libonnxruntime.dylib` inside `Contents/Frameworks/` and sign with `codesign --force --options runtime`).
- Never call ONNX Runtime session creation or file I/O on the real-time audio thread; all session initialization happens inside `Plugin::initialize()` and `assert_process_allocs` guards `Plugin::process()`.
