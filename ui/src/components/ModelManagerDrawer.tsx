import { useCallback, useEffect, useReducer } from 'react';
import type { ModelStatusV1 } from '../api/types';
import { INITIAL_MODEL_MANAGER_STATE, modelManagerReducer } from '../state/machines';
import { getState, useStore } from '../state/store';

const USED_BY_LABEL: Readonly<Record<string, string>> = Object.freeze({
  studio: 'Studio',
  lowband: 'Lowband',
  studio_pro: 'Studio Pro',
  dereverb: 'Dereverb',
  smart_safe: 'Smart Safe MOS',
  restore_source: 'Bandwidth Restoration',
  separate: 'Voice Separation',
});

function formatModelSize(model: ModelStatusV1): string {
  if (model.installed && model.size_bytes > 0) {
    const mb = model.size_bytes / (1024 * 1024);
    return mb >= 10 ? `${Math.round(mb)} MB` : `${mb.toFixed(1)} MB`;
  }
  const approx = model.approx_size_mb;
  return approx >= 10 ? `~${Math.round(approx)} MB` : `~${approx.toFixed(1)} MB`;
}

function formatUsedBy(usedBy: string[]): string {
  return usedBy.map((id) => USED_BY_LABEL[id] ?? id).join(' · ');
}

/** Refresh engine v1 capabilities immediately after installing or verifying a model. */
export async function refreshCapabilities(): Promise<void> {
  const client = getState().client;
  if (!client || typeof client.capabilities !== 'function') return;
  try {
    const res = await client.capabilities();
    if (res && Array.isArray(res.capabilities)) {
      getState().setCapabilitiesV1(res.capabilities);
    }
  } catch {
    /* Non-fatal: periodic health probe will reconcile capabilities on reconnect */
  }
}

export function ModelManagerDrawer() {
  const client = useStore((s) => s.client);
  const engineStatus = useStore((s) => s.engineStatus);
  const setCapabilitiesV1 = useStore((s) => s.setCapabilitiesV1);
  const pushToast = useStore((s) => s.pushToast);

  const [machine, dispatch] = useReducer(modelManagerReducer, INITIAL_MODEL_MANAGER_STATE);
  const models = machine.models;
  const loading = machine.phase === 'loading';
  const loadError = machine.phase === 'error' ? machine.message : null;
  const installingId = machine.phase === 'installing' ? machine.modelId : null;
  const feedbackById = machine.feedbackById;

  const loadModels = useCallback(
    async (signal?: AbortSignal) => {
      if (!client || engineStatus !== 'ready') return;
      dispatch({ type: 'LOAD_START' });
      try {
        const res = await client.getModels(signal);
        if (!signal?.aborted) {
          dispatch({ type: 'LOAD_SUCCESS', models: res.models });
        }
      } catch (err) {
        if (!signal?.aborted) {
          dispatch({
            type: 'LOAD_FAILED',
            message: err instanceof Error ? err.message : 'Failed to inspect local neural models.',
          });
        }
      }
    },
    [client, engineStatus],
  );

  useEffect(() => {
    if (!client || engineStatus !== 'ready') return;
    const ctrl = new AbortController();
    void loadModels(ctrl.signal);
    return () => ctrl.abort();
  }, [client, engineStatus, loadModels]);

  const handleInstall = async (model: ModelStatusV1): Promise<void> => {
    if (!client || installingId !== null) return;
    dispatch({ type: 'INSTALL_START', modelId: model.model_id });
    try {
      const res = await client.installModel(model.model_id);
      if (Array.isArray(res.capabilities) && res.capabilities.length > 0) {
        setCapabilitiesV1(res.capabilities);
      }
      await refreshCapabilities();
      const msg = res.result.message || `${model.display_name} downloaded and SHA-256 verified.`;
      const updatedModels = Array.isArray(res.models) && res.models.length > 0 ? res.models : undefined;
      dispatch({
        type: 'INSTALL_SUCCESS',
        modelId: model.model_id,
        message: msg,
        ...(updatedModels ? { models: updatedModels } : {}),
      });
      if (!updatedModels) {
        await loadModels();
      }
      pushToast(msg, 'ok');
    } catch (err) {
      const msg =
        err instanceof Error
          ? err.message
          : `Failed to download and verify ${model.display_name}.`;
      dispatch({
        type: 'INSTALL_FAILED',
        modelId: model.model_id,
        message: msg,
      });
      pushToast(msg, 'err');
    }
  };

  const verifiedCount = models.filter((m) => m.installed && m.verified).length;

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: 'var(--pad-4)',
        padding: 'var(--pad-5)',
        overflowY: 'auto',
        flex: 1,
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: 'var(--pad-3)',
          color: 'var(--fg-2)',
          fontSize: 'var(--fs-sm)',
        }}
      >
        <p style={{ margin: 0, lineHeight: 1.45 }}>
          Every optional neural weight file is downloaded over HTTPS and verified against its pinned
          SHA-256 digest before activation.
        </p>
        <button
          type="button"
          className="hdr-mode-btn"
          disabled={loading || installingId !== null || engineStatus !== 'ready'}
          onClick={() => void loadModels()}
          aria-label="Refresh neural model status"
          style={{ flexShrink: 0 }}
        >
          {loading ? 'Checking…' : 'Refresh'}
        </button>
      </div>

      {engineStatus !== 'ready' ? (
        <p
          role="status"
          style={{
            margin: 0,
            padding: 'var(--pad-4)',
            borderRadius: 'var(--r)',
            background: 'var(--panel-2)',
            border: '1px solid var(--hair)',
            color: 'var(--fg-2)',
            fontSize: 'var(--fs-sm)',
          }}
        >
          Waiting for the local HawaVoClean engine to connect before listing models.
        </p>
      ) : null}

      {loadError ? (
        <div
          role="alert"
          style={{
            padding: 'var(--pad-4)',
            borderRadius: 'var(--r)',
            background: 'var(--err-bg)',
            border: '1px solid var(--err)',
            color: 'var(--fg)',
            fontSize: 'var(--fs-sm)',
          }}
        >
          {loadError}
        </div>
      ) : null}

      {models.length > 0 ? (
        <p
          role="status"
          aria-live="polite"
          style={{
            margin: 0,
            fontSize: 'var(--fs-label)',
            letterSpacing: 'var(--ls-caps)',
            textTransform: 'uppercase',
            color: 'var(--fg-3)',
          }}
        >
          {verifiedCount} of {models.length} models installed &amp; verified
        </p>
      ) : null}

      <ul
        aria-label="Neural models"
        style={{
          listStyle: 'none',
          margin: 0,
          padding: 0,
          display: 'flex',
          flexDirection: 'column',
          gap: 'var(--pad-4)',
        }}
      >
        {models.map((model) => {
          const isReady = model.installed && model.verified;
          const isTampered = model.installed && !model.verified;
          const isInstalling = installingId === model.model_id;
          const feedback = feedbackById[model.model_id];

          const badgeText = isReady
            ? 'Installed · Verified'
            : isTampered
              ? 'Digest Mismatch'
              : model.downloadable
                ? 'Available to Download'
                : 'Bundled';

          const badgeColor = isReady
            ? 'var(--ok)'
            : isTampered
              ? 'var(--err)'
              : 'var(--amber)';

          return (
            <li
              key={model.model_id}
              data-model-id={model.model_id}
              style={{
                padding: 'var(--pad-4)',
                borderRadius: 'var(--r)',
                background: 'var(--panel-2)',
                border: '1px solid var(--hair)',
                display: 'flex',
                flexDirection: 'column',
                gap: 'var(--pad-3)',
              }}
            >
              <div
                style={{
                  display: 'flex',
                  alignItems: 'flex-start',
                  justifyContent: 'space-between',
                  gap: 'var(--pad-3)',
                }}
              >
                <div style={{ minWidth: 0 }}>
                  <h3
                    style={{
                      margin: 0,
                      fontSize: 'var(--fs-md)',
                      fontWeight: 'var(--fw-medium)',
                      color: 'var(--fg)',
                    }}
                  >
                    {model.display_name}
                  </h3>
                  <div
                    style={{
                      marginTop: '2px',
                      fontSize: 'var(--fs-label)',
                      color: 'var(--fg-3)',
                    }}
                  >
                    {formatUsedBy(model.used_by)}
                  </div>
                </div>
                <span
                  className="badge"
                  style={{
                    flexShrink: 0,
                    color: badgeColor,
                    borderColor: badgeColor,
                  }}
                >
                  {badgeText}
                </span>
              </div>

              <p
                style={{
                  margin: 0,
                  fontSize: 'var(--fs-sm)',
                  color: 'var(--fg-2)',
                  lineHeight: 1.45,
                }}
              >
                {model.description}
              </p>

              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  flexWrap: 'wrap',
                  gap: 'var(--pad-3)',
                  fontSize: 'var(--fs-label)',
                  color: 'var(--fg-3)',
                  fontFamily: 'var(--font-numeric)',
                }}
              >
                <span>
                  License: {model.license} · Size: {formatModelSize(model)}
                  {model.bundled ? ' · Bundled' : ''}
                </span>

                {model.downloadable && !isReady ? (
                  <button
                    type="button"
                    className="hdr-ai-btn"
                    disabled={installingId !== null}
                    aria-busy={isInstalling}
                    onClick={() => void handleInstall(model)}
                    style={{ fontFamily: 'var(--font-sans)' }}
                  >
                    {isInstalling
                      ? 'Downloading & Verifying…'
                      : isTampered
                        ? 'Re-download & Verify'
                        : 'Download & Verify'}
                  </button>
                ) : null}
              </div>

              {isInstalling ? (
                <div
                  role="status"
                  aria-live="polite"
                  style={{
                    fontSize: 'var(--fs-sm)',
                    color: 'var(--cyan-2)',
                  }}
                >
                  Streaming over HTTPS and verifying SHA-256 digest…
                </div>
              ) : null}

              {feedback ? (
                <div
                  role={feedback.kind === 'err' ? 'alert' : 'status'}
                  aria-live={feedback.kind === 'err' ? 'assertive' : 'polite'}
                  style={{
                    fontSize: 'var(--fs-sm)',
                    color: feedback.kind === 'err' ? 'var(--err)' : 'var(--ok)',
                  }}
                >
                  {feedback.text}
                </div>
              ) : null}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
