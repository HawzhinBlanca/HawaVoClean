import fs from 'node:fs';
import path from 'node:path';

export const PACKAGE_PROVENANCE_NAME = 'HAWAVOCLEAN-PACKAGE-PROVENANCE.json';

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/**
 * Permit the hidden runtime self-test only inside the source-bound unsigned
 * app built by package-proof.cjs. A normal signed release has no proof marker;
 * shell-only hosted-CI proofs deliberately carry a false permission.
 */
export function packagedProofSelfTestAllowed(resourcesPath: string, packaged: boolean): boolean {
  if (!packaged) return false;
  const marker = path.join(resourcesPath, PACKAGE_PROVENANCE_NAME);
  try {
    const details = fs.lstatSync(marker);
    if (!details.isFile() || details.isSymbolicLink()) return false;
    const raw: unknown = JSON.parse(fs.readFileSync(marker, 'utf8'));
    if (!isObject(raw) || !isObject(raw.signing)) return false;
    return (
      raw.schema_version === 1 &&
      raw.artifact_type === 'unsigned-macos-app-proof' &&
      raw.distribution_eligible === false &&
      raw.product === 'hawavoclean' &&
      raw.product_version === '3.3.0' &&
      typeof raw.source_revision === 'string' &&
      /^[0-9a-f]{40}$/.test(raw.source_revision) &&
      raw.target === 'macos-arm64' &&
      raw.engine_mode === 'full' &&
      raw.packaged_selftest_allowed === true &&
      raw.signing.developer_id === false &&
      raw.signing.notarized === false &&
      raw.signing.stapled === false
    );
  } catch {
    return false;
  }
}
