"""Byte-identity comparison for published WAV masters.

libsndfile writes a PEAK chunk into every float WAV, and that chunk carries a
Unix **write-time timestamp** (4 bytes at offset 4 of the chunk body). Two
runs of the byte-deterministic pipeline therefore produce byte-identical
files ONLY when they land in the same wall-clock second — a property no test
may bet on (measured: 4 consecutive runs, 3 distinct SHA-256s, every sample
identical, the one differing byte always the PEAK timestamp).

:func:`masked_wav_bytes` zeroes exactly that field so byte-identity tests
compare everything that is actually a product of the pipeline: every header
field, every PEAK level/position, every audio sample. Anything else that
differs still fails the comparison.
"""

import struct


def masked_wav_bytes(data: bytes) -> bytes:
    """Return ``data`` with the PEAK chunk's timestamp field zeroed.

    A non-RIFF payload or a WAV without a PEAK chunk is returned unchanged —
    the comparison then demands strict byte identity.
    """
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return data
    out = bytearray(data)
    pos = 12
    while pos + 8 <= len(data):
        chunk_id = data[pos : pos + 4]
        (size,) = struct.unpack("<I", data[pos + 4 : pos + 8])
        if chunk_id == b"PEAK" and size >= 8 and pos + 16 <= len(data):
            # Chunk body: version (4 bytes), timestamp (4 bytes), then the
            # per-channel peak records. Zero only the timestamp.
            out[pos + 12 : pos + 16] = b"\x00\x00\x00\x00"
        pos += 8 + size + (size & 1)  # chunks are word-aligned
    return bytes(out)
