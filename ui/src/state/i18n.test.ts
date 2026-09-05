import { describe, expect, it, beforeEach } from 'vitest';
import { CATALOGS, getLocale, getTranslation, setLocale, pseudolocalize, type Locale } from './i18n';

describe('i18n and RTL layout catalog', () => {
  beforeEach(() => {
    localStorage.clear();
    setLocale('en');
  });

  it('defaults to English with LTR direction', () => {
    expect(getLocale()).toBe('en');
    expect(document.documentElement.lang).toBe('en');
    expect(document.documentElement.dir).toBe('ltr');
  });

  it('switches to Sorani Kurdish (ckb) with RTL direction and persists to localStorage', () => {
    setLocale('ckb');
    expect(getLocale()).toBe('ckb');
    expect(document.documentElement.lang).toBe('ckb');
    expect(document.documentElement.dir).toBe('rtl');
    expect(localStorage.getItem('hawavoclean_locale')).toBe('ckb');

    const t = getTranslation();
    expect(t.appName).toBe('هاواڤۆکلین');
    expect(t.processButton).toBe('پاککردنەوەی دەنگ');
    expect(t.langToggleLabel).toBe('English');
  });

  it('contains identical translation keys across all supported catalogs', () => {
    const enKeys = Object.keys(CATALOGS.en).sort();
    const ckbKeys = Object.keys(CATALOGS.ckb).sort();
    expect(enKeys).toEqual(ckbKeys);

    for (const key of enKeys) {
      const enVal = CATALOGS.en[key as keyof typeof CATALOGS.en];
      const ckbVal = CATALOGS.ckb[key as keyof typeof CATALOGS.ckb];
      expect(typeof enVal).toBe('string');
      expect(typeof ckbVal).toBe('string');
      expect(enVal.length).toBeGreaterThan(0);
      expect(ckbVal.length).toBeGreaterThan(0);
    }
  });

  it('returns valid fallback translation for unrecognised locale', () => {
    const fallback = getTranslation('unknown' as Locale);
    expect(fallback.appName).toBe('HawaVoClean');
  });

  it('pseudolocalizes text for UI layout stress testing', () => {
    const result = pseudolocalize('Clean Audio');
    expect(result).toBe('[!! Cleeaan AAuudiioo !!]');
  });
});
