# 04 - Training and Data Splits

This page states exactly what the committed checkpoint is, what the training
pipeline now does, and what is still gated on the user. Every claim below is
checkable against the artifacts in this repository.

## What the committed checkpoint actually is

`models/hawarestore-kd/hawarestore_kd.pt` (the checkpoint the restoration
subsystem loads) is an **engineering artifact**, not a Kurdish restoration
model:

- Trained for **3 epochs** on **40 synthetic sine-harmonic items** (the
  simulation generator now in
  `research/restoration/train/train_hawarestore.py`); it has never heard
  Kurdish — or any — real speech.
- Trained with the **flow-matching MSE term only**. The training call at the
  time passed no audio or speaker-embedding arguments, so the STFT, envelope,
  and speaker terms in `research/restoration/train/losses.py` were skipped.
- Used **no split management**: `SplitManager` in
  `research/restoration/train/dataset.py` was never imported by the training
  script that produced it.
- Checkpoint metadata: `epochs=3`, `final_loss=0.078` (flow MSE). It carries
  none of the split/loss metadata described below, which is itself evidence
  of how it was produced.

Its purpose is to prove the inference path end to end: the production loader
(`HawaRestoreKD` in `src/hawavoclean/restoration/hawarestore_kd.py`) refuses
to run on missing or unloadable weights, and this checkpoint exercises that
contract with genuinely trained (if synthetic-only) weights.

## What the training pipeline now does

`research/restoration/train/train_hawarestore.py` (rewritten 2026-08-24) is
ready for the day real Kurdish audio arrives:

### Data modes (explicit, no silent default)
- `--data-dir DIR`: trains on real clean WAV/FLAC files at any sample rate,
  resampled to 48 kHz, with degradations applied on the fly by
  `research/restoration/simulation/degradation.py` (`DegradationSimulator`:
  per-item deterministic cutoff in 2.5-16 kHz, random Butterworth/Chebyshev/
  codec-shape filter, mild additive noise). Speaker attribution comes from
  the `data_dir/<speaker>/<file>` layout (or the filename prefix in flat
  layouts).
- `--synthetic`: the clearly-labeled engineering fallback that generates the
  original sine-harmonic simulation data. The script prints a warning that
  this validates machinery only.
- Passing neither, or both, is an error.

### Speaker-disjoint splits via SplitManager
- Unique speakers are shuffled with a fixed `--split-seed` and partitioned
  into train / validation sets; **no speaker appears in both**.
- Every utterance is registered through
  `SplitManager.add_utterance` (`research/restoration/train/dataset.py`),
  which **raises `ValueError` on any utterance id or content hash appearing
  in more than one split**. (Earlier revisions of this page cited an
  `assert len(train_ids & test_ids) == 0` in `dataset.py`; no such assert
  exists — the raise inside `add_utterance` is the actual mechanism.)
- Manifests (`train.jsonl`, `development.jsonl`, plus the reserved
  calibration/locked splits) are written next to the checkpoint and their
  SHA-256 hashes are recorded in it.

### Full composite loss, active on every step
The trainable loss is `HawaRestoreLoss` in
`research/restoration/train/losses.py`, with all terms live:

| Term | Mechanism | Weight (actual code value) |
|---|---|---|
| Flow | MSE on the flow-matching velocity field | `lambda_flow = 1.0` |
| STFT | Multi-resolution high-band complex STFT (mag + log-mag L1) | `lambda_stft = 1.0` |
| Envelope | Cross-band temporal envelope correlation | `lambda_envelope = 0.5` |
| Speaker | Cosine distance between differentiable log-mel embeddings | `lambda_speaker = 0.2` |

The audio-domain terms are fed by a one-step clean estimate from the linear
probability path (`x1 = (1 - sigma_min) * x_t + (1 - (1 - sigma_min) * t) * v`)
inverted with `torch.istft`, so gradients flow through all four terms. The
trainer hard-fails (`RuntimeError`) if any term fails to report on any step.

Note: `research/restoration/train/loss.py` (`compute_hawarestore_loss`) is a
separate numpy evaluation-side breakdown with its own weights; **the trainer
does not use it**. Earlier revisions of this page attributed the training
weights to that file; the table above quotes the weights that actually train.

### Checkpoint honesty metadata
Every checkpoint saved by the rewritten trainer records: `data_mode`
(`"synthetic"` | `"real"`), `n_train` / `n_val` item counts, `split_seed`,
`train_speakers` / `val_speakers`, `manifest_hashes`, `active_loss_terms`,
`loss_weights`, `epochs`, `final_loss`, and `final_losses` (per-term train
and validation averages of the last epoch). A checkpoint without these fields
was not produced by this pipeline.

### Overwrite protection
The trainer defaults its output to `models/hawarestore-kd-candidate/` and
refuses to overwrite any existing checkpoint without `--overwrite`. The
committed checkpoint at `models/hawarestore-kd/` is replaced only by the
deliberate, user-gated promotion step below.

### Verification
`tests/unit/test_restoration_training.py` runs the pipeline for real (tiny
synthetic run plus a real-WAV-directory run) and asserts: every composite
loss term active and finite, splits speaker-disjoint in both checkpoint
metadata and on-disk manifests, checkpoint reload into `HawaRestoreKDNet`
with the metadata fields, overwrite refusal, and data-mode validation.

## What remains user-gated

1. **Real Kurdish corpus** — collection/licensing of consented Kurdish speech
   is user checkpoint **U3** (`docs/true-10-plan.md`): the user must approve
   every corpus source and its rights before any real-data training run.
2. **Retraining and promotion** — after U3, train with `--data-dir` on the
   approved corpus, evaluate under the locked Sorani protocol
   (`docs/sorani-evaluation-protocol.md`), and only then replace
   `models/hawarestore-kd/hawarestore_kd.pt` with the candidate.
3. Until then, the committed synthetic checkpoint stays in place and this
   page stays honest about what it is.
