// Milestone G0.5: Deliberate failing UI test to prove that a defect in the UI suite
// mechanically turns both 'web, desktop, and Resolve shell' and 'required' red on GitHub Actions.

import { describe, expect, it } from 'vitest';

describe('G0.5 deliberate UI failure injection', () => {
  it('deliberately fails to prove the UI leaf and required aggregate turn red', () => {
    expect(true).toBe(false);
  });
});
