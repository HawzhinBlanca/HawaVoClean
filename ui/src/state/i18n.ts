/**
 * Internationalization (i18n) catalog for English ('en') and Sorani Kurdish ('ckb').
 * Provides RTL layout switching with LTR preservation for waveforms and numeric displays (D4.4).
 */

import { useSyncExternalStore } from 'react';

export type Locale = 'en' | 'ckb';

export interface TranslationCatalog {
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
  engineOffline: string;
  engineConnecting: string;
  engineReady: string;
  engineBusy: string;
  noAnalysis: string;
  profileStudioPro: string;
  profileDereverb: string;
  profileStudio: string;
  profileLowband: string;
  profileProduction: string;
  region: string;
  markStart: string;
  markEnd: string;
  clearRegion: string;
}

export const CATALOGS: Record<Locale, TranslationCatalog> = {
  en: {
    langToggleLabel: 'کوردی',
    langToggleAria: 'کوردی — Switch to Sorani Kurdish',
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
    engineOffline: 'ENGINE OFFLINE',
    engineConnecting: 'ENGINE CONNECTING',
    engineReady: 'ENGINE READY',
    engineBusy: 'ENGINE BUSY',
    noAnalysis: 'No analysis',
    profileStudioPro: 'Studio Pro',
    profileDereverb: 'Dereverb',
    profileStudio: 'Studio',
    profileLowband: 'Low-band',
    profileProduction: 'Production',
    region: 'Region',
    markStart: 'In',
    markEnd: 'Out',
    clearRegion: 'Clear',
  },
  ckb: {
    langToggleLabel: 'English',
    langToggleAria: 'English — گۆڕین بۆ زمانی ئینگلیزی',
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
    engineOffline: 'بزوێنەر لەسەر هێڵ نییە',
    engineConnecting: 'بزوێنەر پەیوەست دەبێت',
    engineReady: 'بزوێنەر ئامادەیە',
    engineBusy: 'بزوێنەر سەرقاڵە',
    noAnalysis: 'بێ شیکاری',
    profileStudioPro: 'ستۆدیۆ پرۆ',
    profileDereverb: 'لابردنی دەنگدانەوە',
    profileStudio: 'ستۆدیۆ',
    profileLowband: 'باندی نزم',
    profileProduction: 'بەرهەمهێنان',
    region: 'ناوچە',
    markStart: 'دەستپێک',
    markEnd: 'کۆتایی',
    clearRegion: 'سڕینەوە',
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
  for (const listener of listeners) {
    listener();
  }
}

export function getLocale(): Locale {
  return currentLocale;
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
    () => 'en' as Locale,
  );

  return {
    locale,
    t: CATALOGS[locale] ?? CATALOGS.en,
    setLocale,
    isRtl: locale === 'ckb',
  };
}
