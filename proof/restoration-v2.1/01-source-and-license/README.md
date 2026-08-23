# Source and License Provenance Matrix (Phase 0)

| Upstream Project | Pinned Commit / Reference | Code License | Weights License | Role in HawaVoClean | Commercial / Production Status |
|---|---|---|---|---|---|
| **UniverSR** (`woongzip1/UniverSR`) | `26dc21c44e11f9f19e823f02b0d4641dd5ea5af2` (arXiv:2510.00771) | MIT | CC-BY-4.0 | Architectural foundation for STFT-domain missing band prediction (vendored at `vendors/universr/`) | Verified. Upstream weights are not used; the committed checkpoint is our own (see note 2). |
| **3D-Speaker** (`modelscope/3D-Speaker`) | `065629c313eaf1a01c65c640c46d77e61e9607b4` (ERes2NetV2) | Apache-2.0 | Apache-2.0 | Design reference only. **No 3D-Speaker code or weights are vendored, imported, or executed anywhere in this repository.** | Candidate for future real-corpus speaker verification after user checkpoint U3. Not integrated. |
| **AnyBand** | arXiv:2608.00572 | N/A (No public code) | N/A | Design reference (continuous cutoffs, frequency-axis modeling, missing-band flow loss) | Architectural reference only; no third-party code copied. |
| **AnyEnhance** | `viewfinder-annn/AnyEnhance-v1` / Amphion `c0277229f83ea685db15611fd81a5396c571e264` | MIT / Academic | Non-Commercial / Unclear | Research comparator considered during design. **No AnyEnhance code or weights are present in this repository.** | Excluded. Never vendored, never loaded. |
| **ClearerVoice-Studio / MossFormer2_SR_48K** | modelscope/ClearerVoice-Studio | Apache-2.0 | ModelScope TOS | Offline research baseline considered during design. Not present in this repository. | Evaluation comparator candidate only. |

## Verification Notes

1. **UniverSR Code**: Licensed under permissive MIT. Adapted into `vendors/universr/` and
   `src/hawavoclean/restoration/universr_upstream.py`.
2. **HawaRestore-KD Checkpoint — training-data truth**: The committed checkpoint
   (`models/hawarestore-kd/hawarestore_kd.pt`) is an **engineering artifact trained on synthetic
   simulation data**. It was produced by `research/restoration/train/train_hawarestore.py`, whose
   `KurdishSimulationDataset` generates sine-harmonic tones with randomized F0 and low-pass
   cutoffs — it contains **no recorded Kurdish speech of any kind**, consented or otherwise, and
   the default run is 3 epochs over 40 one-second synthetic samples. Its purpose is to prove the
   architecture, protected-band mechanics, hashing, and Guard R plumbing end to end. Training on
   a real, licensed, consented Kurdish corpus is scheduled behind user checkpoint U3
   (`docs/true-10-plan.md`), after which this note must be replaced with the real corpus
   provenance record.
3. **Speaker embeddings**: The production speaker-identity metric is
   `src/hawavoclean/restoration/speaker_embed.py`, a **handcrafted deterministic DSP extractor**
   (mel filterbank statistics, formant/timbre features, 192-dim unit-norm vector). It is not
   3D-Speaker, ERes2NetV2, or any neural speaker model. 3D-Speaker remains a referenced candidate
   for post-U3 identity evaluation only.
4. **Fail-Safe Boundary**: AnyEnhance and other unverified external models are not present in
   this repository and cannot enter the production execution path.
