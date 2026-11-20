// True-10 D4.1 · Multi-file batch queue component.
// Displays aggregate progress, per-item status, individual item controls,
// deck inspection without stopping queue, and batch pause/resume/cancel.

import { useMemo, useState } from 'react';
import {
  baseName,
  cancelBatchItem,
  cancelCurrentBatch,
  inspectBatchItem,
  pauseCurrentBatch,
  resumeCurrentBatch,
  retryBatchItem,
} from '../state/actions';
import type { DeliveryPresetId } from '../state/presets';
import { useStore } from '../state/store';
import type { BatchItem } from '../api/types';
import {
  IconCancel,
  IconCheck,
  IconPause,
  IconPlay,
  IconRetry,
  IconWarn,
} from './Icons';
import { JobHistory } from './JobHistory';

const DELIVERY_BADGE_LABEL: Record<DeliveryPresetId, string> = {
  match_input: 'Match Input',
  ebu_r128: 'EBU R128 (-23 LUFS)',
  atsc_a85: 'ATSC A/85 (-24 LUFS)',
  podcast: 'Podcast (-16 LUFS)',
  streaming: 'Streaming (-14 LUFS)',
};

/** Returns true when a batch item has failed, errored, or has reverted/flagged units. */
export function isBatchItemFlagged(item: BatchItem): boolean {
  if (item.state === 'failed' || Boolean(item.error)) return true;
  const s = item.report?.summary;
  if (!s) return false;
  return (
    (s.reverted ?? 0) > 0 ||
    (s.continuity_reverted ?? 0) > 0 ||
    (s.error_passthrough ?? 0) > 0 ||
    (s.unverified ?? 0) > 0
  );
}

function ItemMark({ state }: { state: BatchItem['state'] }) {
  if (state === 'done') return <IconCheck size={12} />;
  if (state === 'failed') return <IconWarn size={12} />;
  if (state === 'cancelled') return <IconCancel size={12} />;
  if (state === 'running') return <span className="batch-spinner" aria-hidden="true" />;
  return <span className="batch-queued-dot" aria-hidden="true" />;
}

function BatchItemRow({
  item,
  isInspecting,
}: {
  item: BatchItem;
  isInspecting: boolean;
}) {
  const name = baseName(item.input_path);
  const pct = Math.round(item.progress * 100);
  const statusText =
    item.state === 'failed'
      ? (item.error?.message || item.message || 'Failed')
      : item.state === 'done'
        ? 'Done'
        : item.state === 'cancelled'
          ? 'Cancelled'
          : `${item.stage || 'Processing'} · ${pct}%`;

  return (
    <div
      className="batch-item-row"
      data-state={item.state}
      data-inspecting={isInspecting ? 'true' : 'false'}
    >
      <div className="batch-item-lead">
        <span className="batch-item-mark" aria-hidden="true">
          <ItemMark state={item.state} />
        </span>
        <div className="batch-item-info">
          <div className="batch-item-title-line">
            <span className="batch-item-name" title={item.input_path}>
              {name}
            </span>
            {isInspecting && <span className="batch-inspect-badge">ON DECK</span>}
            <span className={`batch-state-pill ${item.state}`}>{item.state.toUpperCase()}</span>
          </div>
          <div className="batch-item-sub-line">
            <span className="batch-item-status mono">{statusText}</span>
            {item.state === 'running' && (
              <div
                className="batch-item-progress-track"
                role="progressbar"
                aria-valuenow={pct}
                aria-valuemin={0}
                aria-valuemax={100}
                aria-label={`${item.seq}. ${name} progress: ${pct}%`}
              >
                <div className="batch-item-progress-fill" style={{ width: `${pct}%` }} />
              </div>
            )}
          </div>
        </div>
      </div>
      <div className="batch-item-actions">
        <button
          type="button"
          className="batch-action-btn inspect-btn"
          onClick={() => void inspectBatchItem(item)}
          title={`Inspect ${name} in transport decks without stopping queue`}
          aria-label={`Inspect ${name} on deck`}
        >
          Inspect
        </button>
        {(item.state === 'queued' || item.state === 'running') && (
          <button
            type="button"
            className="batch-action-btn cancel-btn"
            onClick={() => void cancelBatchItem(item.job_id)}
            title={`Cancel ${name}`}
            aria-label={`Cancel ${name}`}
          >
            <IconCancel size={12} />
          </button>
        )}
        {(item.state === 'failed' || item.state === 'cancelled') && (
          <button
            type="button"
            className="batch-action-btn retry-btn"
            onClick={() => void retryBatchItem(item.job_id)}
            title={`Retry ${name}`}
            aria-label={`Retry ${name}`}
          >
            <IconRetry size={12} />
          </button>
        )}
      </div>
    </div>
  );
}

export function BatchQueue() {
  const batch = useStore((s) => s.batch);
  const activeInspectJobId = useStore((s) => s.activeInspectJobId);
  const setBatch = useStore((s) => s.setBatch);
  const batchNamingTemplate = useStore((s) => s.batchNamingTemplate);
  const deliveryPreset = useStore((s) => s.rackSettings.deliveryPreset);

  const [showHistory, setShowHistory] = useState(false);
  const [flaggedOnly, setFlaggedOnly] = useState(false);

  const flaggedJobs = useMemo(
    () => (batch ? batch.jobs.filter(isBatchItemFlagged) : []),
    [batch],
  );

  if (!batch) return null;

  if (showHistory) {
    return (
      <div className="batch-history-wrapper">
        <div className="batch-history-bar">
          <button
            type="button"
            className="batch-btn subtle"
            onClick={() => setShowHistory(false)}
            aria-label="Back to batch queue"
          >
            ← Back to Batch Queue ({batch.completed_items}/{batch.total_items})
          </button>
        </div>
        <JobHistory />
      </div>
    );
  }

  const pct = Math.round(batch.progress * 100);
  const isTerminal =
    batch.state === 'done' || batch.state === 'failed' || batch.state === 'cancelled';
  const isRunning = batch.state === 'running' || batch.state === 'queued';
  const isPaused = batch.state === 'paused';
  const visibleJobs = flaggedOnly ? flaggedJobs : batch.jobs;
  const deliveryLabel = DELIVERY_BADGE_LABEL[deliveryPreset] ?? deliveryPreset;

  return (
    <section className="panel batch-queue" aria-label="Batch Queue">
      <div className="panel-head batch-head">
        <div className="batch-head-left">
          <h2 className="panel-title">
            <span>Batch Queue</span>
            <span className={`batch-state-tag ${batch.state}`}>{batch.state.toUpperCase()}</span>
          </h2>
          <span className="batch-counts mono">
            {batch.completed_items}/{batch.total_items} items ({pct}%)
            {batch.failed_items > 0 && (
              <span className="batch-failed-count"> · {batch.failed_items} failed</span>
            )}
            {batch.cancelled_items > 0 && (
              <span className="batch-cancelled-count"> · {batch.cancelled_items} cancelled</span>
            )}
          </span>
          <div className="batch-meta-bar">
            <span
              className="batch-delivery-badge"
              title={`Active delivery target: ${deliveryLabel}`}
            >
              {deliveryLabel}
            </span>
            <code
              className="batch-naming-badge"
              title={`Batch output naming template: ${batchNamingTemplate}`}
            >
              {batchNamingTemplate}
            </code>
          </div>
        </div>
        <div className="batch-controls">
          <button
            type="button"
            className={`batch-btn subtle batch-flag-toggle${flaggedOnly ? ' on' : ''}`}
            aria-pressed={flaggedOnly}
            aria-label="Flagged only — filter flagged items"
            title="Show only items with errors or reverted speech units"
            onClick={() => setFlaggedOnly((v) => !v)}
          >
            <span>Flagged only</span>
            {flaggedJobs.length > 0 ? (
              <span className="batch-flag-count mono">{flaggedJobs.length}</span>
            ) : null}
          </button>
          <button
            type="button"
            className="batch-btn subtle"
            onClick={() => setShowHistory(true)}
            title="View single-job history runs"
            aria-label="View history"
          >
            <span>History</span>
          </button>
          {isRunning && (
            <button
              type="button"
              className="batch-btn"
              onClick={() => void pauseCurrentBatch()}
              title="Pause remaining items in batch queue"
              aria-label="Pause batch"
            >
              <IconPause size={12} />
              <span>Pause</span>
            </button>
          )}
          {isPaused && (
            <button
              type="button"
              className="batch-btn"
              onClick={() => void resumeCurrentBatch()}
              title="Resume paused batch queue"
              aria-label="Resume batch"
            >
              <IconPlay size={12} />
              <span>Resume</span>
            </button>
          )}
          {!isTerminal && (
            <button
              type="button"
              className="batch-btn danger"
              onClick={() => void cancelCurrentBatch()}
              title="Cancel all remaining items in batch queue"
              aria-label="Cancel batch"
            >
              <IconCancel size={12} />
              <span>Cancel</span>
            </button>
          )}
          {isTerminal && (
            <button
              type="button"
              className="batch-btn subtle"
              onClick={() => setBatch(null)}
              title="Dismiss batch queue view"
              aria-label="Dismiss batch queue"
            >
              <span>Dismiss</span>
            </button>
          )}
        </div>
      </div>

      <div
        className="batch-progress-bar-aggregate"
        role="progressbar"
        aria-valuenow={pct}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={`Batch progress ${pct}%`}
      >
        <div className={`batch-progress-fill ${batch.state}`} style={{ width: `${pct}%` }} />
      </div>

      <div className="batch-body">
        {flaggedOnly && visibleJobs.length === 0 ? (
          <p className="batch-empty-flagged" role="status">
            No flagged items — all runs are clean so far.
          </p>
        ) : (
          <ul className="batch-list" aria-label={`Batch items (${visibleJobs.length})`}>
            {visibleJobs.map((item) => (
              <li key={item.job_id}>
                <BatchItemRow
                  item={item}
                  isInspecting={item.job_id === activeInspectJobId}
                />
              </li>
            ))}
          </ul>
        )}
      </div>
    </section>
  );
}
