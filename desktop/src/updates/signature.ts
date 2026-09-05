import crypto from 'node:crypto';
import type { UpdateManifest } from '../contracts.js';

/**
 * Serialize manifest payload into canonical deterministic JSON for signing.
 * The `signature` property itself is excluded.
 */
export function canonicalizeManifest(manifest: Partial<UpdateManifest>): string {
  const { signature: _sig, ...rest } = manifest;
  const sortedKeys = Object.keys(rest).sort();
  const sortedObj: Record<string, unknown> = {};
  for (const key of sortedKeys) {
    sortedObj[key] = (rest as Record<string, unknown>)[key];
  }
  return JSON.stringify(sortedObj);
}

/**
 * Generate an in-memory Ed25519 keypair formatted as SPKI/PKCS8 PEM strings.
 */
export function generateSigningKeyPair(): { publicKey: string; privateKey: string } {
  const { publicKey, privateKey } = crypto.generateKeyPairSync('ed25519');
  return {
    publicKey: publicKey.export({ type: 'spki', format: 'pem' }).toString(),
    privateKey: privateKey.export({ type: 'pkcs8', format: 'pem' }).toString(),
  };
}

/**
 * Sign an update manifest using an Ed25519 private key.
 * Returns base64 encoded signature.
 */
export function signManifest(
  manifest: Omit<UpdateManifest, 'signature'>,
  privateKeyPem: string | crypto.KeyObject,
): string {
  const data = Buffer.from(canonicalizeManifest(manifest), 'utf8');
  const key = typeof privateKeyPem === 'string'
    ? crypto.createPrivateKey(privateKeyPem)
    : privateKeyPem;
  const signature = crypto.sign(null, data, key);
  return signature.toString('base64');
}

/**
 * Verify an update manifest against an Ed25519 public key.
 * Fails closed on any invalid formatting, tamper, or corrupt signature.
 */
export function verifyManifestSignature(
  manifest: UpdateManifest,
  publicKeyPem: string | crypto.KeyObject,
): { valid: boolean; reason?: string } {
  if (!manifest || typeof manifest !== 'object') {
    return { valid: false, reason: 'invalid_manifest_object' };
  }
  if (!manifest.signature || typeof manifest.signature !== 'string') {
    return { valid: false, reason: 'missing_signature' };
  }

  try {
    const data = Buffer.from(canonicalizeManifest(manifest), 'utf8');
    const signature = Buffer.from(manifest.signature, 'base64');
    const key = typeof publicKeyPem === 'string'
      ? crypto.createPublicKey(publicKeyPem)
      : publicKeyPem;

    const valid = crypto.verify(null, data, key, signature);
    if (!valid) {
      return { valid: false, reason: 'corrupt_signature' };
    }
    return { valid: true };
  } catch (err) {
    return {
      valid: false,
      reason: `corrupt_signature: ${err instanceof Error ? err.message : String(err)}`,
    };
  }
}

/**
 * Strict semver comparator (supports MAJOR.MINOR.PATCH with optional pre-release).
 * Returns:
 *   1 if v1 > v2
 *  -1 if v1 < v2
 *   0 if v1 === v2
 */
export function compareSemver(v1: string, v2: string): number {
  const parse = (v: string) => {
    const clean = v.replace(/^v/, '').trim();
    const [main, pre] = clean.split('-');
    const parts = (main ?? '').split('.').map((p) => {
      const n = parseInt(p, 10);
      return Number.isNaN(n) ? 0 : n;
    });
    while (parts.length < 3) parts.push(0);
    return {
      major: parts[0] ?? 0,
      minor: parts[1] ?? 0,
      patch: parts[2] ?? 0,
      hasPre: Boolean(pre),
      pre: pre ?? '',
    };
  };

  const a = parse(v1);
  const b = parse(v2);

  if (a.major !== b.major) return a.major > b.major ? 1 : -1;
  if (a.minor !== b.minor) return a.minor > b.minor ? 1 : -1;
  if (a.patch !== b.patch) return a.patch > b.patch ? 1 : -1;

  // Release versions rank higher than pre-release versions of the same numeric version
  if (!a.hasPre && b.hasPre) return 1;
  if (a.hasPre && !b.hasPre) return -1;
  if (a.hasPre && b.hasPre) return a.pre.localeCompare(b.pre);

  return 0;
}
