import { useState, type FormEvent } from 'react';
import { useI18n } from '../state/i18n';
import { jobInFlight, useStore } from '../state/store';

export function QuickChatBar() {
  const { locale } = useI18n();
  const engineReady = useStore((s) => s.engineStatus === 'ready');
  const busy = useStore((s) => jobInFlight(s.job) || s.analyzing);
  const setAssistantDrawerOpen = useStore((s) => s.setAssistantDrawerOpen);
  const setPendingChatPrompt = useStore((s) => s.setPendingChatPrompt);
  const [text, setText] = useState('');

  const submit = (e: FormEvent): void => {
    e.preventDefault();
    const trimmed = text.trim();
    if (!trimmed) {
      setAssistantDrawerOpen(true);
      return;
    }
    setPendingChatPrompt(trimmed);
    setAssistantDrawerOpen(true);
    setText('');
  };

  const placeholder =
    locale === 'ckb'
      ? 'بە چات داوای هەرشتێک بکە: نموونە «دەنگدانەوە لاببە، کەمکردنی س، پۆدکاست» یان «کێشەی ئەم دەنگە چییە؟»…'
      : 'Control everything by chat: e.g. "Remove echo, fix harsh S, HPF 90 Hz, master for Podcast" or "What\'s wrong with this audio?"…';

  return (
    <form
      className="quick-chat-bar"
      role="search"
      aria-label="Quick AI audio command bar"
      onSubmit={submit}
    >
      <span className="qcb-badge" aria-hidden="true">
        ✦ Hawa Chat
      </span>
      <label className="sr-only" htmlFor="quick-chat-bar-input">
        Ask Hawa AI to edit, play, diagnose, or export audio
      </label>
      <input
        id="quick-chat-bar-input"
        type="text"
        className="qcb-input"
        value={text}
        maxLength={2000}
        disabled={!engineReady || busy}
        placeholder={placeholder}
        onChange={(e) => setText(e.target.value)}
      />
      <button
        type="submit"
        className="qcb-btn"
        disabled={!engineReady || busy}
        title="Send to Hawa AI Assistant"
      >
        {locale === 'ckb' ? 'جێبەجێکردن' : 'Ask'}
      </button>
    </form>
  );
}
