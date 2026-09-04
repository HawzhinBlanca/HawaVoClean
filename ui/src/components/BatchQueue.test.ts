// components/BatchQueue.test.ts — True-10 D4.1 batch queue UI verification.

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, createElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import type { BatchSummary } from '../api/types';
import { useStore } from '../state/store';
import { BatchQueue } from './BatchQueue';

const mockPause = vi.fn();
const mockResume = vi.fn();
const mockCancelBatch = vi.fn();
const mockCancelItem = vi.fn();
const mockRetryItem = vi.fn();
const mockInspectItem = vi.fn();

vi.mock('../bridge', () => ({
  getBridge: () => ({ host: 'web', engine: { getEndpoint: async () => ({ baseUrl: '' }) } }),
}));

vi.mock('../state/actions', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../state/actions')>()),
  pauseCurrentBatch: () => mockPause(),
  resumeCurrentBatch: () => mockResume(),
  cancelCurrentBatch: () => mockCancelBatch(),
  cancelBatchItem: (id: string) => mockCancelItem(id),
  retryBatchItem: (id: string) => mockRetryItem(id),
  inspectBatchItem: (item: unknown) => mockInspectItem(item),
}));

declare global {
  var IS_REACT_ACT_ENVIRONMENT: boolean | undefined;
}
globalThis.IS_REACT_ACT_ENVIRONMENT = true;

const pristine = useStore.getState();
let host: HTMLElement;
let root: Root;

beforeEach(() => {
  vi.clearAllMocks();
  useStore.setState(pristine, true);
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(async () => {
    root.unmount();
  });
  host.remove();
});

async function render(): Promise<void> {
  await act(async () => {
    root.render(createElement(BatchQueue));
  });
}

function sampleBatch(overrides: Partial<BatchSummary> = {}): BatchSummary {
  return {
    batch_id: 'b_test123',
    state: 'running',
    total_items: 2,
    completed_items: 1,
    failed_items: 0,
    cancelled_items: 0,
    running_items: 1,
    queued_items: 0,
    progress: 0.5,
    created_at: '2026-09-05T00:00:00Z',
    updated_at: '2026-09-05T00:00:10Z',
    jobs: [
      {
        job_id: 'j1',
        seq: 1,
        state: 'done',
        stage: 'done',
        progress: 1.0,
        message: 'Completed',
        input_path: '/path/to/first.wav',
        output_path: '/path/to/first_studio.wav',
        report_path: '/path/to/first_studio.hawavoclean.json',
        profile: 'studio',
        mode: 'natural',
        created_at: '2026-09-05T00:00:00Z',
        started_at: '2026-09-05T00:00:01Z',
        finished_at: '2026-09-05T00:00:05Z',
      },
      {
        job_id: 'j2',
        seq: 2,
        state: 'running',
        stage: 'enhancing',
        progress: 0.4,
        message: 'Processing pass 1',
        input_path: '/path/to/second.mp3',
        output_path: '/path/to/second_studio.wav',
        report_path: '/path/to/second_studio.hawavoclean.json',
        profile: 'studio',
        mode: 'natural',
        created_at: '2026-09-05T00:00:00Z',
        started_at: '2026-09-05T00:00:05Z',
        finished_at: null,
      },
    ],
    ...overrides,
  };
}

describe('BatchQueue component', () => {
  it('renders nothing when batch is null', async () => {
    useStore.getState().setBatch(null);
    await render();
    expect(host.querySelector('.batch-queue')).toBeNull();
  });

  it('renders aggregate batch progress and status', async () => {
    useStore.getState().setBatch(sampleBatch());
    await render();

    const queue = host.querySelector('.batch-queue');
    expect(queue).not.toBeNull();

    const title = host.querySelector('.panel-title');
    expect(title?.textContent).toContain('Batch Queue');
    expect(title?.textContent).toContain('RUNNING');

    const counts = host.querySelector('.batch-counts');
    expect(counts?.textContent).toContain('1/2 items (50%)');

    const progressBar = host.querySelector('.batch-progress-bar-aggregate');
    expect(progressBar?.getAttribute('aria-valuenow')).toBe('50');

    const items = host.querySelectorAll('.batch-item-row');
    expect(items).toHaveLength(2);
    expect(items[0]?.textContent).toContain('first.wav');
    expect(items[1]?.textContent).toContain('second.mp3');
  });

  it('calls pauseCurrentBatch when pause button is clicked', async () => {
    useStore.getState().setBatch(sampleBatch({ state: 'running' }));
    await render();

    const pauseBtn = host.querySelector('button[aria-label="Pause batch"]') as HTMLButtonElement | null;
    expect(pauseBtn).not.toBeNull();
    await act(async () => {
      pauseBtn?.click();
    });
    expect(mockPause).toHaveBeenCalledTimes(1);
  });

  it('calls resumeCurrentBatch when resume button is clicked on paused batch', async () => {
    useStore.getState().setBatch(sampleBatch({ state: 'paused' }));
    await render();

    const resumeBtn = host.querySelector('button[aria-label="Resume batch"]') as HTMLButtonElement | null;
    expect(resumeBtn).not.toBeNull();
    await act(async () => {
      resumeBtn?.click();
    });
    expect(mockResume).toHaveBeenCalledTimes(1);
  });

  it('calls cancelCurrentBatch when cancel batch button is clicked', async () => {
    useStore.getState().setBatch(sampleBatch());
    await render();

    const cancelBtn = host.querySelector('button[aria-label="Cancel batch"]') as HTMLButtonElement | null;
    expect(cancelBtn).not.toBeNull();
    await act(async () => {
      cancelBtn?.click();
    });
    expect(mockCancelBatch).toHaveBeenCalledTimes(1);
  });

  it('calls cancelBatchItem for queued/running items', async () => {
    useStore.getState().setBatch(sampleBatch());
    await render();

    const cancelItemBtn = host.querySelector('button[aria-label="Cancel second.mp3"]') as HTMLButtonElement | null;
    expect(cancelItemBtn).not.toBeNull();
    await act(async () => {
      cancelItemBtn?.click();
    });
    expect(mockCancelItem).toHaveBeenCalledWith('j2');
  });

  it('calls retryBatchItem for failed items', async () => {
    const batchWithFailed = sampleBatch({
      jobs: [
        {
          job_id: 'j3',
          seq: 1,
          state: 'failed',
          stage: 'failed',
          progress: 0,
          message: 'Audio decode error',
          input_path: '/path/to/corrupt.wav',
          output_path: '',
          report_path: '',
          profile: 'production',
          mode: 'natural',
          created_at: '2026-09-05T00:00:00Z',
          started_at: '2026-09-05T00:00:01Z',
          finished_at: '2026-09-05T00:00:02Z',
          error: { code: 'corrupt_media', message: 'Invalid wave header' },
        },
      ],
    });
    useStore.getState().setBatch(batchWithFailed);
    await render();

    const retryBtn = host.querySelector('button[aria-label="Retry corrupt.wav"]') as HTMLButtonElement | null;
    expect(retryBtn).not.toBeNull();
    await act(async () => {
      retryBtn?.click();
    });
    expect(mockRetryItem).toHaveBeenCalledWith('j3');
  });

  it('calls inspectBatchItem without interrupting queue and reflects inspecting status', async () => {
    useStore.getState().setBatch(sampleBatch());
    useStore.getState().setActiveInspectJobId('j1');
    await render();

    const inspectingBadge = host.querySelector('.batch-inspect-badge');
    expect(inspectingBadge).not.toBeNull();
    expect(inspectingBadge?.textContent).toBe('ON DECK');

    const inspectBtn = host.querySelector('button[aria-label="Inspect second.mp3 on deck"]') as HTMLButtonElement | null;
    expect(inspectBtn).not.toBeNull();
    await act(async () => {
      inspectBtn?.click();
    });
    expect(mockInspectItem).toHaveBeenCalledWith(expect.objectContaining({ job_id: 'j2' }));
  });

  it('dismisses completed batch when Dismiss is clicked', async () => {
    useStore.getState().setBatch(sampleBatch({ state: 'done', progress: 1.0, completed_items: 2 }));
    await render();

    const dismissBtn = host.querySelector('button[aria-label="Dismiss batch queue"]') as HTMLButtonElement | null;
    expect(dismissBtn).not.toBeNull();
    await act(async () => {
      dismissBtn?.click();
    });
    expect(useStore.getState().batch).toBeNull();
  });

  it('toggles history view and returns to batch view', async () => {
    useStore.getState().setBatch(sampleBatch());
    await render();

    const historyBtn = host.querySelector('button[aria-label="View history"]') as HTMLButtonElement | null;
    expect(historyBtn).not.toBeNull();

    await act(async () => {
      historyBtn?.click();
    });
    expect(host.querySelector('.batch-history-wrapper')).not.toBeNull();

    const backBtn = host.querySelector('button[aria-label="Back to batch queue"]') as HTMLButtonElement | null;
    expect(backBtn).not.toBeNull();

    await act(async () => {
      backBtn?.click();
    });
    expect(host.querySelector('.batch-history-wrapper')).toBeNull();
    expect(host.querySelector('.batch-queue')).not.toBeNull();
  });
});
