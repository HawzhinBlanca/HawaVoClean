/**
 * Internationalization (i18n) catalog for English ('en') and Sorani Kurdish ('ckb').
 * Provides RTL layout switching with LTR preservation for waveforms and numeric displays.
 */

import { useSyncExternalStore } from 'react';

export type Locale = 'en' | 'ckb';

export interface TranslationCatalog {
  appName: string;
  tagline: string;
  dropPrompt: string;
  dropSubPrompt: string;
  processButton: string;
  processing: string;
  cancel: string;
  original: string;
  cleaned: string;
  restore: string;
  smartSafe: string;
  production: string;
  studio: string;
  preserve: string;
  offlineBanner: string;
  reconnecting: string;
  jobCompleted: string;
  jobFailed: string;
  metricsTitle: string;
  sampleRate: string;
  duration: string;
  channels: string;
  languageName: string;
}

export const CATALOGS: Record<Locale, TranslationCatalog> = {
  en: {
    appName: 'HawaVoClean',
    tagline: 'High-End Kurdish & Multilingual Voice Cleaner',
    dropPrompt: 'Drop audio file here',
    dropSubPrompt: 'WAV, FLAC, MP3, AIFF, or MP4',
    processButton: 'Clean Audio',
    processing: 'Processing...',
    cancel: 'Cancel',
    original: 'Original (A)',
    cleaned: 'Cleaned (B)',
    restore: 'Restore',
    smartSafe: 'Smart Safe',
    production: 'Production',
    studio: 'Studio',
    preserve: 'Preserve',
    offlineBanner: 'Engine offline. Attempting automatic reconnection...',
    reconnecting: 'Reconnecting...',
    jobCompleted: 'Processing complete. Master audio ready.',
    jobFailed: 'Processing failed. Safe fallback engaged.',
    metricsTitle: 'Audio Metrics',
    sampleRate: 'Sample Rate',
    duration: 'Duration',
    channels: 'Channels',
    languageName: 'English',
  },
  ckb: {
    appName: 'هاواڤۆکلین',
    tagline: 'پاککەرەوەی دەنگی پێشکەوتووی کوردی و فرەزمان',
    dropPrompt: 'فایلی دەنگی لێرە دابنێ',
    dropSubPrompt: 'WAV, FLAC, MP3, AIFF, یان MP4',
    processButton: 'پاککردنەوەی دەنگ',
    processing: 'لە جێبەجێکردندایە...',
    cancel: 'پەشیمانبوونەوە',
    original: 'دەنگی بنەڕەتی (A)',
    cleaned: 'دەنگی پاککراوە (B)',
    restore: 'گەڕاندنەوە',
    smartSafe: 'زیرەکی پارێزراو',
    production: 'بەرهەمهێنان',
    studio: 'ستۆدیۆ',
    preserve: 'پاراستن',
    offlineBanner: 'بزوێنەرەکە لەکار کەوتووە. هەوڵی گرێدانەوە دەدرێت...',
    reconnecting: 'دووبارە گرێدانەوە...',
    jobCompleted: 'جێبەجێکردن تەواو بوو. فایلی دەنگی ماستەر ئامادەیە.',
    jobFailed: 'جێبەجێکردن سەرکەوتوو نەبوو. دۆخی پارێزراو چالاک کرا.',
    metricsTitle: 'پێوانەکانی دەنگ',
    sampleRate: 'ڕێژەی نموونە',
    duration: 'ماوە',
    channels: 'کەناڵەکان',
    languageName: 'کوردی (سۆرانی)',
  },
};

let currentLocale: Locale = 'en';
const listeners = new Set<() => void>();

function notify(): void {
  for (const listener of listeners) {
    listener();
  }
}

export function setLocale(locale: Locale): void {
  currentLocale = locale;
  if (typeof document !== 'undefined') {
    document.documentElement.lang = locale;
    document.documentElement.dir = locale === 'ckb' ? 'rtl' : 'ltr';
  }
  notify();
}

export function getLocale(): Locale {
  return currentLocale;
}

export function getTranslation(locale: Locale = currentLocale): TranslationCatalog {
  return CATALOGS[locale] ?? CATALOGS.en;
}

export function useI18n(): {
  locale: Locale;
  t: TranslationCatalog;
  setLocale: (locale: Locale) => void;
  isRtl: boolean;
} {
  const locale = useSyncExternalStore(
    (cb) => {
      listeners.add(cb);
      return () => listeners.delete(cb);
    },
    () => currentLocale,
    () => 'en' as Locale
  );

  return {
    locale,
    t: CATALOGS[locale],
    setLocale,
    isRtl: locale === 'ckb',
  };
}
