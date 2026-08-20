// Engine HTTP API shapes (docs/ui-contract.md §1) and the HawaVoCleanReport
// (docs/schemas/report.schema.json). Only the fields the UI reads are typed
// strictly; everything else is carried through as `unknown`.

export interface HealthResponse {
  ok: boolean;
  version: string;
  profiles: string[];
  engine_pid: number;
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

export type Profile = 'studio' | 'production';

export interface CreateJobRequest {
  input_path: string;
  profile: Profile;
  output_path?: string;
  overwrite?: boolean;
}

export interface CreateJobResponse {
  job_id: string;
  output_path: string;
  report_path: string;
}

export type JobState = 'queued' | 'running' | 'done' | 'failed' | 'cancelled';
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

export interface HawaVoCleanReport {
  schema_version?: number;
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
