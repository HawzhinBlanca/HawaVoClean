# 04 - Training and Data Splits

## Split Strategy and Leakage Prevention
To guarantee scientific validity and zero evaluation contamination:
1. **Speaker-Disjoint Splits**: Test speakers and evaluation utterances are strictly separated from training sets.
2. **Deterministic Hashing**: Every utterance is assigned to `train`, `val`, or `test` via `sha256(canonical_path)` with fixed ratios (80/10/10).
3. **Cross-Split Assertions**:
   - Training manifests: `research/restoration/train/dataset.py` validates `assert len(train_ids & test_ids) == 0`.
   - No speaker prototype audio from test sets is ever used for backbone conditioning during training.

## Training Loss Formulations
Defined in `research/restoration/train/loss.py`:
$$ \mathcal{L}_{\text{total}} = \lambda_{\text{spec}} \mathcal{L}_{\text{multi-res-spec}} + \lambda_{\text{time}} \mathcal{L}_{\text{time}} + \lambda_{\text{f0}} \mathcal{L}_{\text{F0}} + \lambda_{\text{spk}} \mathcal{L}_{\text{speaker-cosine}} + \lambda_{\text{flow}} \mathcal{L}_{\text{velocity}} $$

### Component Weights
- Multi-resolution spectral loss: $\lambda_{\text{spec}} = 1.0$
- Time-domain waveform loss: $\lambda_{\text{time}} = 0.5$
- F0 harmonic alignment loss: $\lambda_{\text{f0}} = 0.3$
- Speaker cosine similarity loss: $\lambda_{\text{spk}} = 0.2$
- Midpoint flow velocity loss: $\lambda_{\text{flow}} = 1.0$
