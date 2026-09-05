// @ts-check
'use strict';

/**
 * DaVinci Resolve WorkflowIntegration.node Preflight Discovery & Validation
 *
 * Enforces legal compliance (consented local discovery from licensed Resolve Studio),
 * Mach-O universal binary architecture validation, and structured diagnostics with
 * exact actionable repair guidance.
 *
 * Implements Task Q5.3 (P0/M).
 */

const fs = require('node:fs');
const path = require('node:path');

const DEFAULT_RESOLVE_APP = '/Applications/DaVinci Resolve/DaVinci Resolve.app';
const DEFAULT_SDK_DIR = '/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Workflow Integrations';
const DEFAULT_SAMPLE_NODE_REL = 'Examples/SamplePlugin/WorkflowIntegration.node';
const DEFAULT_PROMISE_NODE_REL = 'Examples/SamplePromisePlugin/WorkflowIntegration.node';

// Mach-O CPU Types
const CPU_TYPE_X86_64 = 0x01000007;
const CPU_TYPE_ARM64 = 0x0100000c;

// Mach-O Magic Numbers
const FAT_MAGIC = 0xcafebabe;
const FAT_CIGAM = 0xbebafeca;
const MH_MAGIC_64 = 0xfeedfacf;
const MH_CIGAM_64 = 0xcffaedfe;

/**
 * Inspects a binary file to check if it is a valid Mach-O binary and returns contained architectures.
 *
 * @param {string} filePath
 * @returns {{ valid: boolean, isMachO: boolean, architectures: string[], error?: string }}
 */
function inspectMachOBinary(filePath) {
  let fd;
  try {
    const stat = fs.statSync(filePath);
    if (!stat.isFile()) {
      return { valid: false, isMachO: false, architectures: [], error: 'Not a regular file' };
    }
    if (stat.size < 32) {
      return { valid: false, isMachO: false, architectures: [], error: 'File too small for Mach-O header' };
    }

    fd = fs.openSync(filePath, 'r');
    const buf = Buffer.alloc(4096);
    const bytesRead = fs.readSync(fd, buf, 0, 4096, 0);
    if (bytesRead < 32) {
      return { valid: false, isMachO: false, architectures: [], error: 'Header read incomplete' };
    }

    const magic = buf.readUInt32BE(0);
    const archs = [];

    if (magic === FAT_MAGIC || magic === FAT_CIGAM) {
      const isBe = magic === FAT_MAGIC;
      const nfat = isBe ? buf.readUInt32BE(4) : buf.readUInt32LE(4);
      let offset = 8;
      for (let i = 0; i < nfat && offset + 20 <= bytesRead; i++) {
        const cputype = isBe ? buf.readUInt32BE(offset) : buf.readUInt32LE(offset);
        if (cputype === CPU_TYPE_ARM64) archs.push('arm64');
        else if (cputype === CPU_TYPE_X86_64) archs.push('x86_64');
        offset += 20;
      }
      return {
        valid: archs.length > 0,
        isMachO: true,
        architectures: archs,
      };
    } else if (magic === MH_MAGIC_64 || magic === MH_CIGAM_64) {
      const isBe = magic === MH_MAGIC_64;
      const cputype = isBe ? buf.readUInt32BE(4) : buf.readUInt32LE(4);
      if (cputype === CPU_TYPE_ARM64) archs.push('arm64');
      else if (cputype === CPU_TYPE_X86_64) archs.push('x86_64');
      return {
        valid: archs.length > 0,
        isMachO: true,
        architectures: archs,
      };
    }

    return { valid: false, isMachO: false, architectures: [], error: 'Not a Mach-O binary' };
  } catch (err) {
    return { valid: false, isMachO: false, architectures: [], error: err.message };
  } finally {
    if (fd !== undefined) {
      try {
        fs.closeSync(fd);
      } catch (_) {}
    }
  }
}

/**
 * Preflight discovery for DaVinci Resolve Studio Workflow Integrations SDK.
 *
 * @param {object} [options]
 * @param {string} [options.resolveAppPath] Path to DaVinci Resolve.app
 * @param {string} [options.sdkDir] Path to Workflow Integrations SDK directory
 * @param {string} [options.overrideNodePath] Direct override path for testing
 * @param {string} [options.platform] OS platform (defaults to process.platform)
 * @param {string} [options.targetArch] Required architecture (defaults to process.arch)
 * @returns {{
 *   ok: boolean,
 *   status: 'ready' | 'missing_sdk' | 'resolve_not_installed' | 'free_edition_unsupported' | 'invalid_binary' | 'unsupported_platform',
 *   code: string,
 *   path: string | null,
 *   repairGuidance: string | null,
 *   diagnostics: {
 *     platform: string,
 *     hasApp: boolean,
 *     isStudio: boolean,
 *     hasSdkDir: boolean,
 *     nodePathChecked: string | null,
 *     architectures: string[],
 *     fileSizeBytes: number | null
 *   }
 * }}
 */
function discoverWorkflowIntegrationSdk(options = {}) {
  const platform = options.platform || process.platform;
  const targetArch = options.targetArch || (process.arch === 'arm64' ? 'arm64' : 'x86_64');
  const appPath = options.resolveAppPath || DEFAULT_RESOLVE_APP;
  const sdkDir = options.sdkDir || DEFAULT_SDK_DIR;

  const diagnostics = {
    platform,
    hasApp: false,
    isStudio: false,
    hasSdkDir: false,
    nodePathChecked: null,
    architectures: [],
    fileSizeBytes: null,
  };

  if (platform !== 'darwin') {
    return {
      ok: false,
      status: 'unsupported_platform',
      code: 'ERR_UNSUPPORTED_PLATFORM',
      path: null,
      repairGuidance: 'Workflow Integrations plugin qualification is currently supported on macOS (Apple silicon & Intel).',
      diagnostics,
    };
  }

  if (options.overrideNodePath) {
    diagnostics.nodePathChecked = options.overrideNodePath;
    if (fs.existsSync(options.overrideNodePath)) {
      const inspect = inspectMachOBinary(options.overrideNodePath);
      diagnostics.architectures = inspect.architectures;
      try {
        diagnostics.fileSizeBytes = fs.statSync(options.overrideNodePath).size;
      } catch (_) {}

      if (inspect.valid && inspect.architectures.includes(targetArch)) {
        return {
          ok: true,
          status: 'ready',
          code: 'SDK_READY',
          path: options.overrideNodePath,
          repairGuidance: null,
          diagnostics,
        };
      }
      return {
        ok: false,
        status: 'invalid_binary',
        code: 'ERR_SDK_INVALID_BINARY',
        path: options.overrideNodePath,
        repairGuidance: `WorkflowIntegration.node at ${options.overrideNodePath} is invalid or lacks ${targetArch} architecture. Found: [${inspect.architectures.join(', ')}].`,
        diagnostics,
      };
    }
  }

  // 1. Check DaVinci Resolve application existence
  const hasApp = fs.existsSync(appPath);
  diagnostics.hasApp = hasApp;

  // Check if DaVinci Resolve is Studio vs Free
  // Resolve Studio has Workflow Integrations Developer directory or DaVinci Resolve Studio in Info.plist / bundle name
  let isStudio = false;
  if (hasApp) {
    try {
      const plistPath = path.join(appPath, 'Contents', 'Info.plist');
      if (fs.existsSync(plistPath)) {
        const plistContent = fs.readFileSync(plistPath, 'utf8');
        if (plistContent.includes('Studio') || plistContent.includes('com.blackmagic-design.DaVinciResolveStudio')) {
          isStudio = true;
        }
      }
    } catch (_) {}
  }

  // 2. Check Workflow Integrations Developer SDK directory
  const hasSdkDir = fs.existsSync(sdkDir);
  diagnostics.hasSdkDir = hasSdkDir;
  if (hasSdkDir) {
    // If the Developer SDK directory exists, Studio is definitely present
    isStudio = true;
  }
  diagnostics.isStudio = isStudio;

  if (!hasApp && !hasSdkDir) {
    return {
      ok: false,
      status: 'resolve_not_installed',
      code: 'ERR_RESOLVE_NOT_INSTALLED',
      path: null,
      repairGuidance: `DaVinci Resolve Studio was not detected at '${appPath}'. Please install DaVinci Resolve Studio 20.1 or newer before installing the workflow integration plugin.`,
      diagnostics,
    };
  }

  // If App exists, but Developer SDK is missing:
  if (hasApp && !hasSdkDir) {
    if (!isStudio) {
      return {
        ok: false,
        status: 'free_edition_unsupported',
        code: 'ERR_RESOLVE_FREE_EDITION',
        path: null,
        repairGuidance:
          'DaVinci Resolve (Free Edition) does not support Workflow Integration plugins. Workflow Integrations require DaVinci Resolve Studio 20.1 or newer. To clean dialogue without Resolve Studio, launch the standalone HawaVoClean desktop application (/Applications/HawaVoClean.app).',
        diagnostics,
      };
    }

    return {
      ok: false,
      status: 'missing_sdk',
      code: 'ERR_SDK_MISSING',
      path: null,
      repairGuidance: `DaVinci Resolve Studio is installed, but the Developer SDK directory is missing at '${sdkDir}'. Re-run the DaVinci Resolve Studio installer and ensure Developer tools are checked.`,
      diagnostics,
    };
  }

  // 3. Locate WorkflowIntegration.node
  const candidatePaths = [
    path.join(sdkDir, DEFAULT_SAMPLE_NODE_REL),
    path.join(sdkDir, DEFAULT_PROMISE_NODE_REL),
  ];

  let resolvedPath = null;
  for (const cand of candidatePaths) {
    if (fs.existsSync(cand)) {
      resolvedPath = cand;
      break;
    }
  }

  diagnostics.nodePathChecked = resolvedPath;

  if (!resolvedPath) {
    return {
      ok: false,
      status: 'missing_sdk',
      code: 'ERR_SDK_NODE_NOT_FOUND',
      path: null,
      repairGuidance: `WorkflowIntegration.node was not found in Developer SDK examples at '${sdkDir}'. Expected at '${candidatePaths[0]}'.`,
      diagnostics,
    };
  }

  // 4. Validate Mach-O architecture
  try {
    diagnostics.fileSizeBytes = fs.statSync(resolvedPath).size;
  } catch (_) {}

  const inspect = inspectMachOBinary(resolvedPath);
  diagnostics.architectures = inspect.architectures;

  if (!inspect.valid || !inspect.architectures.includes(targetArch)) {
    return {
      ok: false,
      status: 'invalid_binary',
      code: 'ERR_SDK_INVALID_BINARY',
      path: resolvedPath,
      repairGuidance: `WorkflowIntegration.node at '${resolvedPath}' is invalid or incompatible with architecture '${targetArch}'. Found architectures: [${inspect.architectures.join(', ')}].`,
      diagnostics,
    };
  }

  return {
    ok: true,
    status: 'ready',
    code: 'SDK_READY',
    path: resolvedPath,
    repairGuidance: null,
    diagnostics,
  };
}

/**
 * Consented acquisition of WorkflowIntegration.node into the plugin stage.
 *
 * @param {string} targetDir Directory where WorkflowIntegration.node should be copied/linked
 * @param {object} [options] Preflight discovery options
 * @returns {{ ok: boolean, installedPath?: string, error?: string, code?: string }}
 */
function acquireWorkflowIntegrationNode(targetDir, options = {}) {
  const discovery = discoverWorkflowIntegrationSdk(options);
  if (!discovery.ok || !discovery.path) {
    return {
      ok: false,
      code: discovery.code,
      error: discovery.repairGuidance || 'SDK discovery failed',
    };
  }

  const destPath = path.join(targetDir, 'WorkflowIntegration.node');
  try {
    fs.mkdirSync(targetDir, { recursive: true });
    // Copy the file atomically
    const tempDest = `${destPath}.tmp.${Date.now()}`;
    fs.copyFileSync(discovery.path, tempDest);
    fs.chmodSync(tempDest, 0o755);
    fs.renameSync(tempDest, destPath);

    return {
      ok: true,
      installedPath: destPath,
      code: 'ACQUIRED_OK',
    };
  } catch (err) {
    return {
      ok: false,
      code: 'ERR_ACQUISITION_FAILED',
      error: `Failed to copy WorkflowIntegration.node to ${destPath}: ${err.message}`,
    };
  }
}

module.exports = {
  CPU_TYPE_ARM64,
  CPU_TYPE_X86_64,
  DEFAULT_RESOLVE_APP,
  DEFAULT_SDK_DIR,
  inspectMachOBinary,
  discoverWorkflowIntegrationSdk,
  acquireWorkflowIntegrationNode,
};
