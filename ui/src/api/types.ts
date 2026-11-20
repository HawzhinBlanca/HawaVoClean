// Engine HTTP API shapes (docs/ui-contract.md §1) and the HawaVoCleanReport
// (docs/schemas/report.schema.json). Only the fields the UI reads are typed
// strictly; everything else is carried through as `unknown`.

export type DeliveryPresetId =
  | 'match_input'
  | 'ebu_r128'
  | 'atsc_a85'
  | 'podcast'
  | 'streaming';

export type ExportFormatId = 'wav24' | 'wav32f' | 'flac' | 'aac' | 'video_mux';

export interface HealthResponse {
  ok: boolean;
  version: string;
  profiles: string[];
  engine_pid: number;
  /**
   * Restore capability (docs/ui-contract.md, Addendum 2): the sorted speaker
   * ids whose profiles the engine can load, recomputed per request, and the
   * one flag the UI keys the restore control's existence on. Optional because
   * a contract-revision-1 engine never sends them — absence reads exactly
   * like `restore_available: false`, and the control stays hidden.
   */
  speakers?: string[];
  restore_available?: boolean;
}

export interface AudioAnalysis {
  path: string;
  duration_s: number;
  sample_rate: number;
  channels: number;
  peaks: { min: number[]; max: number[] };
  rms_db: number[];
  spectrum: { freqs_hz: number[]; db: number[] };
  loudness: { integrated_lufs: number | null; true_peak_dbtp: number | null };
  noise_floor_db: number | null;
  /** True only when ffprobe detected a real video track (excludes audio-only .mp4 / cover art). */
  has_video?: boolean;
}

/**
 * `POST /api/peaks` — windowed waveform (docs/ui-contract.md, Addendum 1).
 * Same semantics as the `/api/analyze` waveform fields, computed over
 * [start_s, end_s] only, so deep zoom shows real detail instead of
 * interpolated whole-file buckets.
 */
export interface PeaksRequest {
  path: string;
  start_s: number;
  end_s: number;
  buckets?: number;
}

export interface PeaksWindow {
  path: string;
  start_s: number;
  end_s: number;
  sample_rate: number;
  channels: number;
  duration_s: number;
  /** 1 means one sample per bucket — there is no more detail to fetch. */
  samples_per_bucket: number;
  peaks: { min: number[]; max: number[] };
  rms_db: number[];
}

export type Profile = 'studio_pro' | 'dereverb' | 'studio' | 'lowband' | 'production';
export type CleanupProfile = Profile;

/** Per-job processing mode (docs/ui-contract.md, Addendum 2, True-10 D4.11). */
export type JobMode = 'natural' | 'restore' | 'smart_safe';

export type CapabilityMaturity = 'qualified' | 'experimental' | 'blocked';

export interface CapabilityStatusV1 {
  capability_id: string;
  available: boolean;
  maturity: CapabilityMaturity;
  reason?: string | null;
  manifest_sha256?: string | null;
  providers?: string[];
}

export interface CapabilitiesResponseV1 {
  schema_version: 1;
  capabilities: CapabilityStatusV1[];
}

export type RestorePolicy = 'disabled' | 'source_allowed' | 'enrolled_only' | 'auto';

export interface SmartSafeStrategyV1 {
  kind: 'smart_safe';
  restore_policy?: RestorePolicy;
  restorePolicy?: RestorePolicy;
  speaker_profile_id?: string | null;
  speakerProfileId?: string | null;
  allow_generative_reconstruction: boolean;
  allowGenerativeReconstruction?: boolean;
  strength?: number | undefined;
  ambience_floor_db?: number | null | undefined;
  ambienceFloorDb?: number | null | undefined;
  delivery_preset?: DeliveryPresetId | null | undefined;
  deliveryPreset?: DeliveryPresetId | null | undefined;
  stereo_mode?: 'auto' | 'dual_mono_same' | 'split_speakers' | 'true_stereo_ms' | null | undefined;
  stereoMode?: 'auto' | 'dual_mono_same' | 'split_speakers' | 'true_stereo_ms' | null | undefined;
  hpf_enabled?: boolean | undefined;
  hpfEnabled?: boolean | undefined;
  hpf_cutoff_hz?: number | undefined;
  hpfCutoffHz?: number | undefined;
  pro_repair?: Record<string, boolean> | null | undefined;
  proRepair?: Record<string, boolean> | null | undefined;
}

export type ManualRoute =
  | 'production'
  | 'studio'
  | 'studio_pro'
  | 'dereverb'
  | 'lowband'
  | 'lowband_then_production'
  | 'restore_source'
  | 'restore_enrolled';

export interface ManualStrategyV1 {
  kind: 'manual';
  route: ManualRoute;
  speaker_profile_id?: string | null | undefined;
  speakerProfileId?: string | null | undefined;
  expert_cutoff_hz?: number | null | undefined;
  expertCutoffHz?: number | null | undefined;
  allow_generative_reconstruction: boolean;
  allowGenerativeReconstruction?: boolean | undefined;
  strength?: number | undefined;
  time_range?: [number, number] | null | undefined;
  timeRange?: [number, number] | null | undefined;
  ambience_floor_db?: number | null | undefined;
  ambienceFloorDb?: number | null | undefined;
  delivery_preset?: DeliveryPresetId | null | undefined;
  deliveryPreset?: DeliveryPresetId | null | undefined;
  stereo_mode?: 'auto' | 'dual_mono_same' | 'split_speakers' | 'true_stereo_ms' | null | undefined;
  stereoMode?: 'auto' | 'dual_mono_same' | 'split_speakers' | 'true_stereo_ms' | null | undefined;
  hpf_enabled?: boolean | undefined;
  hpfEnabled?: boolean | undefined;
  hpf_cutoff_hz?: number | undefined;
  hpfCutoffHz?: number | undefined;
  pro_repair?: Record<string, boolean> | null | undefined;
  proRepair?: Record<string, boolean> | null | undefined;
}

export interface SeparateStrategyV1 {
  kind: 'separate';
  operation: 'extract' | 'remove';
  prompt: string;
  model_size: 'small' | 'base' | 'large';
  quality: 'high';
  seed: number;
  cleanup_profile: CleanupProfile | null;
  device: 'auto' | 'cpu' | 'mps' | 'cuda';
  strength?: number | undefined;
  time_range?: [number, number] | null | undefined;
  timeRange?: [number, number] | null | undefined;
}

export type ProcessingStrategyV1 = SmartSafeStrategyV1 | ManualStrategyV1 | SeparateStrategyV1;

export interface AudioChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface AudioTimeRange {
  start_s: number;
  end_s: number;
}

export type BandwidthShape = 'unknown' | 'silence' | 'fullband' | 'steep_lowpass';

export interface AcousticContext {
  duration_s?: number | undefined;
  sample_rate?: number | undefined;
  channels?: number | undefined;
  noise_risk?: number | undefined;
  reverberation_risk?: number | undefined;
  music_risk?: number | undefined;
  hum_confidence?: number | undefined;
  rumble_confidence?: number | undefined;
  clipping_risk?: number | undefined;
  estimated_cutoff_hz?: number | undefined;
  bandwidth_shape?: BandwidthShape | undefined;
  noise_floor_db?: number | undefined;
  integrated_lufs?: number | undefined;
  available_profiles?: CleanupProfile[] | undefined;
  separation_available?: boolean | undefined;
}

export interface AudioPlanRequest {
  message: string;
  history: AudioChatMessage[];
  allow_cloud_text: boolean;
  acoustic_context?: AcousticContext | null | undefined;
}

export interface AudioPlan {
  operation: 'clean' | 'restore' | 'extract' | 'remove' | 'clarify';
  target: string | null;
  cleanup_profile: CleanupProfile | null;
  model_size: 'small' | 'base' | 'large';
  quality: 'high';
  strength?: number | undefined;
  time_range?: AudioTimeRange | null | undefined;
  explanation: string;
  provider: 'rules' | 'ollama' | 'openai';
  needs_clarification: boolean;
  fallback_reason?: string | null;
}

export interface AudioAssistantStatus {
  provider: 'auto' | 'rules' | 'ollama' | 'openai' | 'unconfigured';
  available?: boolean;
  reason?: string | null;
  cloud_text_available?: boolean;
  local_model?: string | null;
  ollama_reachable?: boolean;
  installed_models?: string[];
  active_mode?: 'local_llm' | 'deterministic_rules';
}

export interface ProcessingRequestV1 {
  schema_version?: 1 | undefined;
  schemaVersion?: 1 | undefined;
  source_ids: string[];
  sourceIds?: string[] | undefined;
  strategy: ProcessingStrategyV1;
  execution_policy?: 'offline_only' | 'prefer_offline' | 'cloud_allowed' | undefined;
  executionPolicy?: 'offline_only' | 'prefer_offline' | 'cloud_allowed' | undefined;
  cloud_consent_id?: string | null | undefined;
  cloudConsentId?: string | null | undefined;
  conflict_policy?: 'unique' | 'fail' | 'replace' | undefined;
  conflictPolicy?: 'unique' | 'fail' | 'replace' | undefined;
  record_bundle?: boolean | undefined;
  recordBundle?: boolean | undefined;
  idempotency_key?: string | undefined;
  idempotencyKey?: string | undefined;
}

export interface CreateV1JobItem {
  sourceId: string;
  jobId: string;
  outputPath: string;
  reportPath: string;
  recordBundle?: boolean;
  bundlePath?: string;
  bundle?: unknown;
}

export interface CreateV1JobResponse {
  schemaVersion: 1;
  batchId?: string;
  execution?: string;
  jobs: CreateV1JobItem[];
}

export interface UploadResponse {
  path: string;
  source_id?: string;
}

export interface CreateJobRequest {
  input_path: string;
  profile: Profile;
  output_path?: string;
  overwrite?: boolean;
  /**
   * Restore-mode fields (Addendum 2). The engine pins this request
   * `extra="forbid"` and refuses `speaker_id`/`cutoff_hz` outside restore
   * mode with a 422 — so a natural-mode submit must *omit* all three, never
   * send null placeholders. `speaker_id` is required with `mode: 'restore'`;
   * `cutoff_hz` absent means the cutoff is auto-detected.
   */
  mode?: JobMode;
  speaker_id?: string;
  cutoff_hz?: number;
}

export interface CreateJobResponse {
  job_id: string;
  output_path: string;
  report_path: string;
}

export type JobState = 'queued' | 'running' | 'done' | 'failed' | 'cancelled' | 'interrupted';
export type JobStage =
  | 'preflight'
  | 'decode'
  | 'segment'
  | 'enhance'
  | 'guard'
  | 'finish'
  | 'publish'
  | 'done'
  | 'error';

export interface JobError {
  code: string;
  message: string;
}

export interface JobStatus {
  job_id: string;
  state: JobState;
  stage: JobStage;
  progress: number;
  message: string;
  unit?: { index: number; total: number } | null;
  input_path: string;
  output_path: string;
  report_path: string;
  profile: Profile;
  /**
   * Always present in a contract-revision-2 snapshot; optional here so a
   * revision-1 snapshot (or a cached one) still types. `speaker_id` and
   * `cutoff_hz` appear only when `mode` is `'restore'` — natural snapshots
   * are byte-compatible with revision 1 — and `cutoff_hz` is null when the
   * cutoff was auto-detected.
   */
  mode?: JobMode | 'separate';
  speaker_id?: string;
  cutoff_hz?: number | null;
  started_at: string | null;
  finished_at: string | null;
  error?: JobError | null;
  report?: HawaVoCleanReport | null;
}

export interface ApiError {
  error: string;
  message?: string;
}

/**
 * The engine publishes three artefacts side by side — `<stem>.wav`,
 * `<stem>.hawavoclean.json` and `<stem>.hawavoclean.txt` (see
 * `job.publish_atomically`) — but only the first two appear in `JobStatus`.
 * The human-readable sidecar is the report path with its extension swapped,
 * which is a convention of the publisher, not a guess.
 */
export function reportTxtPath(reportPath: string): string {
  return reportPath.replace(/\.json$/i, '.txt');
}

// ---- HawaVoCleanReport -----------------------------------------------------

export type GuardScoreValue = number | string | boolean;

export interface MediaStats {
  path: string;
  sha256: string;
  sample_rate: number;
  channels: number;
  samples: number;
  duration_s: number;
  integrated_lufs?: number | null;
  true_peak_dbtp?: number | null;
}

export interface UnitDecisionRecord {
  unit_id: number;
  channel: number;
  start_sample: number;
  end_sample: number;
  start_time_s: number;
  end_time_s: number;
  is_speech: boolean;
  input_sha256: string;
  candidate_sha256?: string | null;
  output_sha256: string;
  guard_a_verdict: string;
  guard_a_scores?: Record<string, GuardScoreValue>;
  guard_b_verdict?: string | null;
  guard_b_scores?: Record<string, GuardScoreValue>;
  chosen_strength?: number;
  finish_preset_applied?: string;
  finish_actions?: string[];
  final_decision: string;
  manual_override?: string | null | undefined;
  decision_reason?: string;
  runtime_ms?: number;
}

export interface UnitSummary {
  units_total?: number;
  enhanced?: number;
  reverted?: number;
  unverified?: number;
  error_passthrough?: number;
  continuity_reverted?: number;
  /** Enhanced units that kept their enhancement across a forced mid-speech cut
   *  by fading back to the original at the joint, rather than being reverted
   *  whole. These units are `enhanced` in `final_decision`; the count is here
   *  so the panel can say what a run paid for continuity. */
  continuity_crossfaded?: number;
  no_speech?: number;
  finish_applied?: number;
  finish_bypassed?: number;
}

export interface ReviewTimecode {
  unit_id: number;
  start_time_s: number;
  end_time_s: number;
  channel: number;
  verdict: string;
  reason: string;
}

/** Release identity a schema-v2 report carries (`report.schema.py`). */
export interface ReleaseMetadata {
  product: string;
  version: string;
  report_schema_version: number;
  identity_sha256: string;
}

/** One pass of a multi-pass run. Only the fields the UI presents are typed. */
export interface PassRecord {
  pass_index: number;
  input_sha256: string;
  output_sha256: string;
  units_total?: number;
  enhanced?: number;
  reverted?: number;
  chosen_strengths?: number[];
  separation_db: number;
  integrated_lufs?: number | null;
  discarded?: boolean;
  discard_reason?: string | null;
}

// ---- restoration section (schema v2, docs/ui-contract.md Addendum 2) -------
// Field names mirror `hawavoclean.restoration.report.RestorationReport`; the
// engine serialises the dataclass verbatim, so these are wire names.

export interface RestorationBandwidthEvidence {
  spectral_rolloff?: number;
  above_cutoff_snr_db?: number;
  stationarity?: number;
  high_band_energy_ratio_db?: number;
}

export interface RestorationBandwidth {
  effective_cutoff_hz: number;
  confidence: number;
  /** "codec_lowpass" | "steep_brickwall" | "gentle_rolloff" | "fullband" | "manual_override" | … */
  shape: string;
  restore_recommended?: boolean;
  /** "auto" (measured) or "manual" (asserted by the operator). */
  cutoff_mode?: string;
  evidence?: RestorationBandwidthEvidence;
}

export interface RestorationRestorer {
  name?: string;
  commit?: string;
  weights_sha256?: string;
  checkpoint_path?: string;
  device?: string;
  seed_policy?: string;
  solver?: string;
  steps?: number;
  guidance_scale?: number;
}

export interface RestorationSegments {
  restored: number;
  reduced: number;
  reverted: number;
  bypassed: number;
  errors: number;
}

export interface RestorationGuardR {
  /** "PASS" | "WARN" | "FAIL" | "ERROR" | "NO_RESTORE" */
  verdict?: string;
  accepted_strength?: number;
  reason?: string;
  protected_band?: Record<string, unknown>;
  ctc?: Record<string, unknown>;
  highband_events?: Record<string, unknown>;
  harmonic?: Record<string, unknown>;
  speaker?: Record<string, unknown>;
}

export interface CandidateEvaluationRecord {
  route: string;
  score: number;
  confidence: number;
  cost?: number | undefined;
  rank?: number | undefined;
  selected?: boolean | undefined;
  status: 'accepted' | 'rejected' | 'fallback' | string;
  rejection_reason?: string | null | undefined;
}

export interface AcousticDetectionsRecord {
  speech?: number;
  music_risk?: number;
  crosstalk?: number;
  cutoff_hz?: number | null;
  noise_floor_db?: number | null;
  snr_db?: number | null;
  rt60_s?: number | null;
  clipping_fraction?: number;
  coherence?: number;
}

export interface RestorationSection {
  mode: string;
  speaker_id: string | null;
  profile_hash?: string | null;
  natural_output_hash?: string | null;
  bandwidth?: RestorationBandwidth;
  restorer?: RestorationRestorer;
  segments?: RestorationSegments;
  guard_r?: RestorationGuardR;
  review_timecodes?: unknown[];
  post_mastering_verification?: Record<string, unknown>;
  // Smart Safe decision details (True-10 D4.11 / I3.8)
  selected_route?: string;
  confidence?: number;
  abstained?: boolean;
  reason?: string;
  decision_sha256?: string;
  ranker_version?: string;
  ranker_sha256?: string;
  fallback_route?: string | null;
  candidates?: CandidateEvaluationRecord[];
  detections?: AcousticDetectionsRecord;
}

export interface HawaVoCleanReport {
  schema_version?: number;
  /** Schema v2 only; a schema-v1 report never carries either. */
  release?: ReleaseMetadata | null;
  build?: Record<string, unknown> | null;
  job_id: string;
  config_hash: string;
  input: MediaStats;
  output: MediaStats;
  core: { id: string; algorithm: string; params_hash: string; phase_coherent?: boolean };
  guard: { id: string; probe_hash: string; calibration_id: string };
  environment: Record<string, unknown>;
  summary: UnitSummary;
  review_timecodes?: ReviewTimecode[];
  units?: UnitDecisionRecord[];
  /** Multi-pass audit trail; empty for the ordinary single-pass run. */
  passes?: PassRecord[];
  /** Present only on a restore-mode run (Addendum 2). */
  restoration?: RestorationSection | null;
  /** Signal validation for separation is not a speech-content guarantee. */
  separation?: {
    operation: 'extract' | 'remove';
    prompt: string;
    model_size?: string;
    qualification?: string;
    [key: string]: unknown;
  } | null;
}

// Verdict classes used by the verdict strip. The engine's `final_decision`
// vocabulary: enhanced | original_reverted | original_continuity |
// original_unverified | original_error | original_no_speech.
export type VerdictClass = 'enhanced' | 'reverted' | 'passthrough' | 'error';

export function classifyDecision(decision: string): VerdictClass {
  const d = decision.toLowerCase();
  if (d === 'enhanced') return 'enhanced';
  if (d.includes('error')) return 'error';
  if (d.includes('revert') || d.includes('continuity') || d.includes('unverified')) return 'reverted';
  return 'passthrough';
}

/**
 * Guard R verdicts reuse the verdict-strip classes so the restoration card
 * speaks the same colour language as the per-unit verdicts. FAIL maps to
 * `reverted`, not `error`: a failed restoration *shipped the Natural master*,
 * which is the guard doing its job, not the run breaking. Only ERROR — the
 * restorer itself falling over — earns the error class.
 */
export function classifyRestorationVerdict(verdict: string | null | undefined): VerdictClass {
  const v = (verdict ?? '').toUpperCase();
  if (v === 'PASS' || v === 'WARN') return 'enhanced';
  if (v === 'ERROR') return 'error';
  if (v === 'FAIL') return 'reverted';
  return 'passthrough'; // NO_RESTORE and anything unrecognised: Natural shipped untouched
}

export function decisionLabel(decision: string): string {
  switch (decision) {
    case 'enhanced':
      return 'ENHANCED';
    case 'original_reverted':
      return 'REVERTED';
    case 'original_continuity':
      return 'CONTINUITY REVERT';
    case 'original_unverified':
      return 'UNVERIFIED';
    case 'original_error':
      return 'ERROR PASSTHROUGH';
    case 'original_no_speech':
      return 'NO SPEECH';
    default:
      return decision.replace(/_/g, ' ').toUpperCase();
  }
}

// ---- Durable Batch Types (True-10 E1.2 / D4.1) ------------------------------

export interface BatchItem {
  job_id: string;
  seq: number;
  state: 'queued' | 'running' | 'done' | 'failed' | 'cancelled' | 'interrupted';
  stage: string;
  progress: number;
  message: string;
  input_path: string;
  output_path: string;
  report_path: string;
  profile: Profile;
  mode: JobMode;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  error?: JobError | null;
  report?: HawaVoCleanReport | null;
}

export interface BatchSummary {
  batch_id: string;
  state: 'queued' | 'running' | 'paused' | 'done' | 'failed' | 'cancelled';
  total_items: number;
  completed_items: number;
  failed_items: number;
  cancelled_items: number;
  running_items: number;
  queued_items: number;
  progress: number;
  created_at: string;
  updated_at: string;
  jobs: BatchItem[];
}

// ---- Pro Studio API Types (Wave 2) -----------------------------------------

export type SpectrogramMode = 'original' | 'cleaned' | 'delta';
export type SpectrogramScale = 'log' | 'mel' | 'linear';

export interface SpectrogramTileRequest {
  path: string;
  cleanedPath?: string | null | undefined;
  mode: SpectrogramMode;
  startS: number;
  endS: number;
  widthCols?: number | undefined;
  nBins?: number | undefined;
  nFft?: number | undefined;
  scale?: SpectrogramScale | undefined;
  dbFloor?: number | undefined;
  dbCeil?: number | undefined;
}

export interface SpectrogramTileResponse {
  start_s: number;
  end_s: number;
  sample_rate: number;
  f_min_hz: number;
  f_max_hz: number;
  n_bins: number;
  width_cols: number;
  scale: SpectrogramScale;
  db_floor: number;
  db_ceil: number;
  data_b64: string;
}

export interface RegionPreviewRequest {
  path: string;
  startS: number;
  endS: number;
  profile?: Profile | string | undefined;
  strength?: number | undefined;
  ambienceFloorDb?: number | null | undefined;
  hpfEnabled?: boolean | undefined;
  hpfCutoffHz?: number | undefined;
  deliveryPreset?: DeliveryPresetId | null | undefined;
  stereoMode?: 'auto' | 'dual_mono_same' | 'split_speakers' | 'true_stereo_ms' | null | undefined;
  proRepair?: Record<string, boolean> | null | undefined;
}

export interface RegionPreviewMetrics {
  lufs: number | null;
  noiseFloorDb: number | null;
  mos?: number | null | undefined;
}

export interface RegionPreviewResponse {
  previewPath: string;
  audioUrl: string;
  renderMs: number;
  warmHit: boolean;
  metricsBefore: RegionPreviewMetrics;
  metricsAfter: RegionPreviewMetrics;
  durationS: number;
}

export type UnitOverrideDecision = 'auto' | 'force_original' | 'force_enhanced' | 'custom_strength';

export interface UnitOverrideRequest {
  unitId: number;
  decision: UnitOverrideDecision;
  strength?: number | undefined;
}

export interface UnitOverrideResponse {
  jobId: string;
  unitIndex: number;
  unitId: number;
  oldDecision: string;
  newDecision: string;
  override: UnitOverrideDecision;
  restitched: boolean;
  restitchMs?: number | null | undefined;
  masterSha256?: string | null | undefined;
  summary?: UnitSummary | null | undefined;
  units?: UnitDecisionRecord[] | null | undefined;
}

export interface DeliveryExportRequest {
  masterPath: string;
  destPath?: string | null | undefined;
  format: ExportFormatId;
  deliveryPreset?: DeliveryPresetId | undefined;
  sourceMediaPath?: string | null | undefined;
  preserveMetadata?: boolean | undefined;
}

export interface DeliveryExportResponse {
  outputPath: string;
  format: ExportFormatId;
  deliveryPreset: DeliveryPresetId;
  lraLu: number | null;
  integratedLufs: number | null;
  truePeakDbtp: number | null;
}

export interface TrackInfo {
  index: number;
  name: string;
  sample_rate: number;
  samples: number;
  duration_s: number;
  rms_dbfs: number;
  peak_dbfs: number;
  speech_activity_ratio: number;
  likely_dialogue: boolean;
}

export interface TrackInspectResponse {
  path: string;
  trackCount: number;
  tracks: TrackInfo[];
}

export interface TrackExtractRequest {
  path: string;
  trackIndices: number[];
  outDir?: string | undefined;
}

export interface TrackExtractResponse {
  path?: string | undefined;
  outDir?: string | undefined;
  extractedPaths: string[];
  sourceIds?: string[] | undefined;
}

export interface SpectralRepairRequest {
  path: string;
  startS: number;
  endS: number;
  fLowHz: number;
  fHighHz: number;
  attenuationDb?: number | undefined;
}

export interface SpectralRepairResponse {
  outputPath: string;
  audioUrl: string;
  startS: number;
  endS: number;
  fLowHz: number;
  fHighHz: number;
  attenuationDb: number;
  renderMs: number;
}

// ---- Hash-Verified Model Manager API Types (Wave E) ------------------------

export interface ModelStatusV1 {
  model_id: string;
  display_name: string;
  description: string;
  used_by: string[];
  license: string;
  expected_sha256: string;
  approx_size_mb: number;
  bundled: boolean;
  downloadable: boolean;
  installed: boolean;
  verified: boolean;
  size_bytes: number;
  path?: string | null | undefined;
}

export interface ModelsResponseV1 {
  schema_version: 1;
  models: ModelStatusV1[];
}

export interface ModelInstallResultV1 {
  model_id: string;
  installed: boolean;
  verified: boolean;
  sha256: string;
  size_bytes: number;
  path?: string | null | undefined;
  already_installed: boolean;
  message: string;
}

export interface ModelInstallResponseV1 {
  schema_version: 1;
  ok: boolean;
  result: ModelInstallResultV1;
  models: ModelStatusV1[];
  capabilities?: CapabilityStatusV1[] | undefined;
}


