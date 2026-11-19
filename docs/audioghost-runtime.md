# Optional AudioGhost runtime

HawaVoClean's separation adapter executes AudioGhost in a separate local Python
environment. It does not install models, accept Hugging Face terms, download
assets or silently substitute the existing cleaner for a failed separation.

Set these **engine-process environment variables**, never client-supplied paths:

```sh
export HAWAVOCLEAN_AUDIOGHOST_PYTHON=/absolute/path/to/audioghost-env/bin/python
export HAWAVOCLEAN_AUDIOGHOST_ROOT=/absolute/path/to/audioghost-ai
```

For a source checkout, `python scripts/install_audioghost_runtime.py` prints a
reviewable installation plan. Add `--install` to install only the isolated
runtime under `.local-ai/`, without downloading SAM weights or accepting gated
model terms. Its bounded local manifest is discovered automatically when both
environment overrides are absent. If using overrides, set both together.
The installer can reuse an existing FFmpeg 4–8 shared-library directory via
`--ffmpeg-lib`; it does not downgrade the system FFmpeg. The library directory
is applied only to the isolated model worker. For explicit runtime overrides,
set `HAWAVOCLEAN_AUDIOGHOST_FFMPEG_LIB` as needed. TorchCodec 0.9.1 cannot use
the system's FFmpeg 9 libraries; on this machine compatible FFmpeg 8 libraries
already exist in the user's Miniconda installation.

The checkout must use AudioGhost commit
`79653d1a789f4fc73893d10733e2324ed088cecf`. The three imported upstream worker
files are checked against that commit's Git blob identities before execution.
AudioGhost is MIT-licensed; its separate SAM-Audio code/model dependencies use
Meta's SAM License and require the applicable model-access approval.

Use upstream's Python 3.11 installation instructions and tested constraints:
torch/torchaudio 2.9.0, NumPy below 2, transformers below 5, huggingface_hub below
1, with SAM-Audio installed from git at
`bb4c6999d2677c7402360e426afc01ddfad6dce0` (the adapter verifies its installed
distribution's commit metadata). The parent's newer numerical environment is
not changed. After separately
obtaining model access and downloading the selected SAM checkpoint, its
configuration and T5 encoder/tokenizer assets, start HawaVoClean with the two
variables set. Inference and readiness probes force Hugging Face and Transformers
offline mode. No credential is passed in the worker arguments or request.

Check installation from HawaVoClean with:

```sh
python -m hawavoclean.separation.installation --model-size small --device cpu
```

Add `--check-access` to make a read-only online metadata request using an
already-configured Hugging Face credential. This reports only access status,
never the token. It does not accept license terms or download model files.
Exit status 2 means the optional runtime remains unavailable.

After the owner receives access on the relevant Meta model page, authenticate
locally using the isolated runtime's installed CLI (both `hf` and the older
`huggingface-cli` entry points were verified present):

```sh
.local-ai/audioghost-env/bin/hf auth login
.venv/bin/python scripts/download_audioghost_model.py --check
.venv/bin/python scripts/download_audioghost_model.py --model-size large
```

Enter the credential only in the local login prompt. The explicit download
command obtains only the selected Meta configuration/checkpoint and the T5
configuration, tokenizer and necessary PyTorch or safetensors weights. It does
not accept gated terms, download unused vision/ranking models, run inference or
certify output quality. Use `--model-size small` for a smaller initial setup.
The command is never invoked by processing or readiness checks.

The API accepts a decoded WAV, one explicit sound prompt, Extract or Remove,
Small/Base/Large, Fast/Balanced/High, a fixed seed and a local provider. CPU is the
default. CUDA is optional; MPS is experimental and explicitly selected. CPU/MPS
use FP32, CUDA FP16. High requests 32 upstream integration steps; it is not a
measured guarantee of best perceptual quality. Remove selects the model residual
rather than subtracting the target from the original.

The initial contract is at most two minutes, one or two channels, and 8–192 kHz
WAV. Full input, noise and chunk results remain in upstream host memory; long
media is not qualified. Model input is resampled to the installed codec's rate;
the selected output is resampled back and independently checked for exact source
rate, channel count, sample count, finite samples and extreme amplitudes. Silence
is valid when an intentionally removed target occupied the whole recording.

The parent owns the complete worker process tree, cancellation and a deadline.
Failed and cancelled operations publish no selected stem. Final output creation
is exclusive and cannot replace an existing file. Returned metadata includes
the exact SAM checkpoint/configuration and text encoder SHA-256, selected stem,
provider/precision, seed/settings, code revision and experimental status.
The actual isolated runtime's package versions are recorded separately from
the main engine's finishing environment.

Source review and adapter tests establish the orchestration and validation
boundary. They do **not** establish real AudioGhost output quality, Sorani word
preservation, MPS reliability, six-hour support or a packaged runtime installation.
No SAM-Audio cached models were found on this machine during implementation.

Implementation check on 19 November 2026: the isolated Python 3.11.15 runtime
was installed successfully with torch/torchaudio 2.9.0, NumPy 1.26.4,
transformers 4.57.6, huggingface_hub 0.36.2, and the pinned SAM source commit.
The upstream engine and SAM processor import successfully with the compatible
FFmpeg 8 shared libraries. A full offline worker probe stops specifically on
missing SAM/text-encoder assets; ordinary readiness returns that unavailable
state in about 1 ms without importing the large runtime. No Hugging Face
credential is configured, and no model inference was run.

Sources checked 19 November 2026:
[AudioGhost pinned engine](https://github.com/0x0funky/audioghost-ai/blob/79653d1a789f4fc73893d10733e2324ed088cecf/backend/workers/sam_lite.py),
[tested dependencies](https://github.com/0x0funky/audioghost-ai/blob/79653d1a789f4fc73893d10733e2324ed088cecf/backend/constraints.txt),
[model access and SAM License](https://huggingface.co/facebook/sam-audio-small).
