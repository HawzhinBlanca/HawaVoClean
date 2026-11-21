# Disk-backed Natural rendering

This document describes the bounded-memory Natural rendering boundary. It is
an implementation record, not evidence that the three-hour release gate has
passed.

## Production route

Every Natural job decodes through `iter_decode_audio` into a planar float32
memory map. The decoder has an independent six-hour sample ceiling, a
no-progress timeout, bounded pipe buffering, complete process-tree cleanup and
incremental disk-space reservation. Container duration metadata is not used to
decide whether decode may allocate the complete file in memory.

The exact decoded frame count selects the downstream route:

- Below 64 MiB decoded PCM, existing Natural DSP, mastering and encoding stay
  on the byte-pinned short-file path. The decoded source itself remains
  disk-backed and independently bounded.
- At or above 64 MiB decoded PCM, speech masks are omitted, channel analysis is
  reduced in chunks, enhancement requests are admitted in batches of eight,
  selected and finished units are persisted to disk, assembly writes into a
  caller-owned memory map, and mastering/encoding operate in chunks.

Long Natural jobs admit at most two isolated model workers and at most eight
pending unit requests per dispatch batch. This is a fixed concurrency boundary
in addition to the worker pool's machine-level memory heuristic.

Consumed stages are flushed, closed and unlinked before the next file-length
stage is created. Finishing overwrites the selected-unit stage only after
continuity and file-level tonal analysis have consumed the selected signals,
so the highest steady scratch point is two raw float32 copies. Planarization
also transiently needs two raw copies. The pipeline rechecks free space from
the actual decoded length before entering the long route and retains a 500 MiB
safety margin.

Mastering preserves the existing algorithm:

1. A streaming BS.1770/true-peak analysis pass computes the static gain.
2. The canonical look-ahead limiter writes the gained master to a disk-backed
   stage while carrying release state across chunks.
3. The required post-master loudness/true-peak verification reads that stage,
   then the WAV encoder streams deterministic PCM24 dither or float32 samples.
   Classic RIFF is used while its 32-bit size field is safe; larger masters
   use self-contained RF64. The wall-clock timestamp libsndfile adds to float
   PEAK chunks is normalized to zero so sample-identical renders hash alike.

The final verification is deliberately not folded into encoding: a failed
ceiling or metric calculation must fail closed before publication.

## Equivalence contract

The allocating and disk-backed implementations share the same assembly seam
maths. Streaming loudness matches the existing pyloudnorm operand order,
limiter chunks include real FIR and future-envelope context, and each encoded
channel carries one persistent deterministic RNG. The tests pin:

- decoded samples and source metrics;
- segmentation boundaries, speech decisions and input hashes with and without
  retained masks;
- caller-owned assembly, limiter output, PCM24 dither and float WAV bytes
  across awkward chunk boundaries;
- canonical hashes for non-contiguous arrays;
- bounded and allocating channel-classification outcomes on representative
  dual-mono and split-speaker inputs;
- an end-to-end Natural fixture rendered once below and once above the
  threshold, with identical master bytes, metrics, summary and unit hashes;
- fail-closed cleanup when scratch capacity disappears during decode.

The focused suite lives in
`tests/unit/test_streaming_natural_render.py`; the continuity integration test
also captures the disk-decoded source and preserves its exact seam invariant.

## What still scales with duration

Audio-size heap allocations are removed from the long Natural route, but these
smaller structures still grow with recording length or acoustic fragmentation:

- VAD frame energies and activity flags (one record per 10 ms hop);
- speech interval/unit metadata, decision records and the final JSON report;
- BS.1770 block-energy records (approximately ten blocks per second);
- disk-backed PCM stages and operating-system page cache;
- the output WAV and immutable published generation.

Guard, alignment and finishing allocations are bounded by one configured
speech unit rather than by the complete recording. The maximum resident set
therefore also depends on worker count, model/runtime memory and OS page-cache
accounting, not just these Python arrays.

Restore is outside this boundary and still contains whole-file decode,
assembly/resampling and restoration allocations. No Restore long-audio claim
is implied.

## Unclosed release evidence

The three-hour 48 kHz stereo `<2 GB RSS` acceptance gate remains **unclaimed**.
It requires a real Natural render on both release-target hosts (M1/16 GB macOS
and the supported 8-core/16 GB Windows machine), with peak RSS, scratch high
water, runtime, cancellation responsiveness, output verification and artifact
hashes captured. VAD throughput and memory-map/page-cache behaviour must be
measured in those runs. Until that evidence exists, long-audio capability must
remain blocked in release qualification even though the production code path
is bounded by construction and exercised on short deterministic fixtures.
