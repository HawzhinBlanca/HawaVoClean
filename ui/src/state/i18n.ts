/**
 * Internationalization (i18n) catalog for English ('en') and Sorani Kurdish ('ckb').
 * Provides RTL layout switching with LTR preservation for waveforms and numeric displays (D4.4).
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
  langToggleLabel: string;
  langToggleAria: string;
  noClip: string;
  uploading: string;
  clip: string;
  sent: string;
  analyzing: string;
  running: string;
  stage: string;
  unit: string;
  units: string;
  done: string;
  failed: string;
  reason: string;
  cancelled: string;
  armed: string;
  length: string;
  format: string;
  mono: string;
  stereo: string;
  channelsSuffix: string;
  noise: string;
  lufs: string;
  shortcutsPrompt: string;
  engineOffline: string;
  engineConnecting: string;
  engineReady: string;
  engineBusy: string;
  noAnalysis: string;
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
    langToggleLabel: 'کوردی',
    langToggleAria: 'Switch to Sorani Kurdish',
    noClip: 'No clip',
    uploading: 'Uploading',
    clip: 'Clip',
    sent: 'Sent',
    analyzing: 'Analyzing',
    running: 'Running',
    stage: 'Stage',
    unit: 'Unit',
    units: 'Units',
    done: 'Done',
    failed: 'Failed',
    reason: 'Reason',
    cancelled: 'Cancelled',
    armed: 'Armed',
    length: 'Length',
    format: 'Format',
    mono: 'mono',
    stereo: 'stereo',
    channelsSuffix: 'ch',
    noise: 'Noise',
    lufs: 'LUFS',
    shortcutsPrompt: 'Keyboard Shortcuts',
    engineOffline: 'ENGINE OFFLINE',
    engineConnecting: 'ENGINE CONNECTING',
    engineReady: 'ENGINE READY',
    engineBusy: 'ENGINE BUSY',
    noAnalysis: 'No analysis',
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
    langToggleLabel: 'English',
    langToggleAria: 'گۆڕین بۆ زمانی ئینگلیزی',
    noClip: 'هیچ کلیپێک نییە',
    uploading: 'بارکردن...',
    clip: 'کلیپ',
    sent: 'نێردراو',
    analyzing: 'شیکردنەوە...',
    running: 'کارپێکردن...',
    stage: 'قۆناغ',
    unit: 'یەکە',
    units: 'یەکەکان',
    done: 'تەواو',
    failed: 'شکستی هێنا',
    reason: 'هۆکار',
    cancelled: 'هەڵوەشێنرایەوە',
    armed: 'ئامادەکراو',
    length: 'درێژی',
    format: 'شێواز',
    mono: 'مۆنۆ',
    stereo: 'ستێریۆ',
    channelsSuffix: 'کەناڵ',
    noise: 'ژاوەژاو',
    lufs: 'LUFS',
    shortcutsPrompt: 'کورتەبڕەکانی تەختەکلیل',
    engineOffline: 'بزوێنەر لەسەر هێڵ نییە',
    engineConnecting: 'بزوێنەر پەیوەست دەبێت',
    engineReady: 'بزوێنەر ئامادەیە',
    engineBusy: 'بزوێنەر سەرقاڵە',
    noAnalysis: 'بێ شیکاری',
  },
};

const STORAGE_KEY = 'hawavoclean_locale';

function getInitialLocale(): Locale {
  if (typeof window !== 'undefined' && window.localStorage) {
    try {
      const stored = window.localStorage.getItem(STORAGE_KEY);
      if (stored === 'en' || stored === 'ckb') {
        return stored;
      }
    } catch {
      // Ignore storage access restrictions
    }
  }
  return 'en';
}

let currentLocale: Locale = getInitialLocale();
const listeners = new Set<() => void>();

function notify(): void {
  for (const listener of listeners) {
    listener();
  }
}

export function setLocale(locale: Locale): void {
  currentLocale = locale;
  if (typeof window !== 'undefined' && window.localStorage) {
    try {
      window.localStorage.setItem(STORAGE_KEY, locale);
    } catch {
      // Ignore storage access restrictions
    }
  }
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

/**
 * Pseudolocalizes a string for layout testing: expands vowels and wraps in delimiters
 * to verify UI layout doesn't truncate or wrap unexpectedly.
 */
export function pseudolocalize(text: string): string {
  const transformed = text.replace(/[aeiouAEIOU]/g, (char) => `${char}${char}`);
  return `[!! ${transformed} !!]`;
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
    t: CATALOGS[locale] ?? CATALOGS.en,
    setLocale,
    isRtl: locale === 'ckb',
  };
}
