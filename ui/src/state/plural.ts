// One string family, one rule. "1 of 1 units enhanced" shipped because every
// place that says how a run went pluralised by hand — the screen-reader live
// region, the footer status line, the run list's spoken sentence, the
// copy-summary one-liner and the verdict strip each carried their own
// template, and the plural fix only ever reached the one a bug was filed on.
// The rule is the ordinary English one: the noun agrees with the count it
// follows — "N units" agrees with N, "X of Y units enhanced" agrees with Y
// (0 of 1 unit, 1 of 1 unit, 2 of 2 units).

/** `unit` when the count it names is exactly 1, else `units`. */
export function unitNoun(count: number): 'unit' | 'units' {
  return count === 1 ? 'unit' : 'units';
}

/** `5/5 units enhanced` · `1/1 unit enhanced` — the compact written form the
 *  footer, the run list and the copy-summary line share. */
export function unitsEnhanced(enhanced: number, total: number): string {
  return `${enhanced}/${total} ${unitNoun(total)} enhanced`;
}

/** `5 of 5 units enhanced` · `1 of 1 unit enhanced` — the spoken form the
 *  screen-reader live region carries. */
export function unitsEnhancedSpoken(enhanced: number, total: number): string {
  return `${enhanced} of ${total} ${unitNoun(total)} enhanced`;
}
