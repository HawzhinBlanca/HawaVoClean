// The plural family. "1 of 1 units enhanced" shipped because five templates
// pluralised by hand and the fix only ever reached the one a bug was filed
// on; the helper is now the single place the rule lives, so these tests are
// the rule's whole statement.

import { describe, expect, it } from 'vitest';
import { unitNoun, unitsEnhanced, unitsEnhancedSpoken } from './plural';

describe('unitNoun', () => {
  it('is singular only at exactly one', () => {
    expect(unitNoun(0)).toBe('units');
    expect(unitNoun(1)).toBe('unit');
    expect(unitNoun(2)).toBe('units');
    expect(unitNoun(5)).toBe('units');
  });
});

describe('unitsEnhanced (the written form)', () => {
  it('agrees with the total, not the enhanced count', () => {
    expect(unitsEnhanced(1, 1)).toBe('1/1 unit enhanced');
    expect(unitsEnhanced(0, 1)).toBe('0/1 unit enhanced');
    expect(unitsEnhanced(1, 2)).toBe('1/2 units enhanced');
    expect(unitsEnhanced(5, 5)).toBe('5/5 units enhanced');
    expect(unitsEnhanced(0, 0)).toBe('0/0 units enhanced');
  });
});

describe('unitsEnhancedSpoken (the live-region form)', () => {
  it('agrees with the total, not the enhanced count', () => {
    expect(unitsEnhancedSpoken(1, 1)).toBe('1 of 1 unit enhanced');
    expect(unitsEnhancedSpoken(0, 1)).toBe('0 of 1 unit enhanced');
    expect(unitsEnhancedSpoken(5, 5)).toBe('5 of 5 units enhanced');
  });
});
