# Production media preflight

The Natural processing boundary accepts these input families:

- WAV (`.wav`), AIFF (`.aif`, `.aiff`, `.aifc`) and FLAC (`.flac`)
- MP3 (`.mp3`)
- M4A (`.m4a`) and MP4 audio extraction (`.mp4`, including names such as
  `recording.m4a.mp4`)

Both the filename extension and the probed container must agree. Renamed,
extensionless, malformed and unsupported containers fail closed before the
whole-file render begins. MP4 video is not rewritten; only its selected audio
stream is processed.

Production limits are 8 GiB on-disk size, six hours of audio, one or two
channels, and the sample-rate envelope configured for the selected profile.
The probe rejects non-finite or contradictory duration/sample metadata,
excessive stream tables, extreme metadata integers, invalid stream indices and
unsupported channel layouts. Refusals use `MediaPreflightError.reason`, a
stable `MediaPreflightReason` value suitable for client explanations.

Decoded samples pass the decoder's NaN, infinity and abnormal-amplitude
checks before DSP. Long inputs at or above 64 MiB decoded PCM use the
disk-backed streaming Natural pipeline (`docs/natural-streaming-render.md`),
which batches unit processing, avoids speech mask allocations, and assembles
and masters in memory-mapped chunks to enforce bounded memory below 2 GB RSS.
