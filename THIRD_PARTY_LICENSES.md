# Third-Party Licenses

This distribution vendors or depends on the following third-party works.
Verbatim license texts are reproduced below as their terms require.

---

## DeepFilterNet3 (model weights, vendored)

- Upstream: https://github.com/Rikorose/DeepFilterNet
- Vendored files: `src/voiceclean/resources/models/deepfilternet3/`
  (config.ini, checkpoints/model_120.ckpt.best), hash-locked in
  `studio-core.lock.toml`.
- License: MIT (upstream offers MIT OR Apache-2.0; this distribution takes
  the MIT option).

The MIT License (MIT)
Copyright (c) 2021 Hendrik Schröter

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

## nara_wpe (dependency, not vendored)

- Upstream: https://github.com/fgnt/nara_wpe
- Used for single-channel WPE dereverberation in the studio core.

MIT License

Copyright (c) 2018 Communications Engineering Group, Paderborn University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
---

## Other dependencies

numpy (BSD-3-Clause), scipy (BSD-3-Clause), soundfile (BSD-3-Clause),
pyloudnorm (MIT), pydantic (MIT), and — studio extra only — torch
(BSD-3-Clause), torchaudio (BSD-2-Clause), deepfilternet (MIT OR
Apache-2.0). These are installed from PyPI, not vendored; their license
texts ship with their packages.
