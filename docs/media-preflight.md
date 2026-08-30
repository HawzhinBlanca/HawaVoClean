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

Decoded samples still pass the decoder's NaN, infinity and abnormal-amplitude
checks before DSP. This boundary does **not** make the Natural renderer
streaming: the current render path still decodes the complete audio stream into
memory. Six-hour, bounded-memory production qualification therefore remains
open until the renderer and mastering stages are converted and tested as a
streaming pipeline.
