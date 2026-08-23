# Source and License Provenance Matrix (Phase 0)

| Upstream Project | Pinned Commit / Reference | Code License | Weights License | Role in HawaVoClean | Commercial / Production Status |
|---|---|---|---|---|---|
| **UniverSR** (`woongzip1/UniverSR`) | `26dc21c44e11f9f19e823f02b0d4641dd5ea5af2` (arXiv:2510.00771) | MIT | CC-BY-4.0 | Architectural foundation for STFT-domain missing band prediction | Verified. Training our own Kurdish weights under clean license. |
| **3D-Speaker** (`modelscope/3D-Speaker`) | `065629c313eaf1a01c65c640c46d77e61e9607b4` (ERes2NetV2) | Apache-2.0 | Apache-2.0 | Reference prototype feature extractor and speaker verification metric | Verified for speaker embeddings & identity evaluation. |
| **AnyBand** | arXiv:2608.00572 | N/A (No public code) | N/A | Design reference (continuous cutoffs, frequency-axis modeling, missing-band flow loss) | Architectural reference only; no third-party code copied. |
| **AnyEnhance** | `viewfinder-annn/AnyEnhance-v1` / Amphion `c0277229f83ea685db15611fd81a5396c571e264` | MIT / Academic | Non-Commercial / Unclear | Isolated research comparator only. Forbidden in production path. | Quarantined. Never loaded in production. |
| **ClearerVoice-Studio / MossFormer2_SR_48K** | modelscope/ClearerVoice-Studio | Apache-2.0 | ModelScope TOS | Offline research baseline | Evaluation comparator only. |

## Verification Notes
1. **UniverSR Code**: Licensed under permissive MIT. Adapted into `vendors/universr/` and `src/hawavoclean/restoration/universr_upstream.py`.
2. **HawaRestore-KD Weights**: Fresh Kurdish-trained backbone weights trained exclusively on consented, clean Kurdish dialogue. No synthetic TTS clones used as clean references.
3. **Fail-Safe Boundary**: AnyEnhance and unverified external models are strictly quarantined from the production execution path.
