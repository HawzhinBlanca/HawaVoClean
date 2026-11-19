# Local audio chat

The audio assistant plans a single edit for review. It never reads your audio or
starts processing by itself. Cloud chat is disabled in this implementation.

Local AI uses Ollama's structured output API, followed by application validation.
If local AI is unavailable, the default `auto` provider understands common English
commands and labels the resulting plan as a limited-command fallback. Ambiguous
or unsupported requests require clarification.

## Start the local model service

After Ollama is installed and `qwen3:8b` has been downloaded to this checkout's
`.local-ai/ollama-models`, run this from the repository root:

```sh
OLLAMA_MODELS="$PWD/.local-ai/ollama-models" \
OLLAMA_HOST=127.0.0.1:11435 OLLAMA_NO_CLOUD=1 ollama serve
```

This uses a separate port from the usual Ollama service. It enables local
inference only. It does not download a model or enable cloud inference. Stop
the service with Ctrl-C when it is no longer needed.

The source checkout discovers this optional, non-secret installer manifest:

```json
{"model":"qwen3:8b","url":"http://127.0.0.1:11435"}
```

Save it as `.local-ai/chat-runtime.json`. The local cache/configuration is
ignored by Git and must remain separate from `.hawavoclean-work`, which is
owned by processing and test cleanup.

Environment overrides take precedence: `HAWAVOCLEAN_CHAT_MODEL`,
`HAWAVOCLEAN_OLLAMA_URL` (a loopback HTTP base address), and
`HAWAVOCLEAN_CHAT_PROVIDER` (`auto`, `ollama`, or `rules`). Without a manifest
or overrides, the planner tries model `default` at `http://127.0.0.1:11434`.
The planner never starts a model service automatically.

Only text instructions/history are sent to the local model. Responses use
Large/High separation settings unless faster processing is explicitly requested.
The model unloads after planning so separation can use the available memory.

Structured responses are not proof that an instruction was understood correctly.
Review the described operation and listen to the result before exporting.

## Verification on this machine

On 19 November 2026, the installed `qwen3:8b` passed a 13-case synthetic
instruction check through the real local service and planner. This exercised
speech/non-speech extraction, sound removal, dialogue cleanup, follow-up changes,
and clarification of unsupported or ambiguous requests. Two product-limit cases
use deterministic guards; an explicit isolation-intent correction is also
labelled as a rules result. Model requests took approximately 2–3 seconds after
installation, with a 6.1 GiB total allocation reported by Ollama on Apple M5 Max.

The model manifest SHA-256 was
`500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41`.
Detailed local proofs are `.local-ai/planner-live-verification.json` and
`.local-ai/planner-paraphrase-verification.json`. The first paraphrase failures
are preserved separately as `.local-ai/planner-paraphrase-before-fix.json`.

These checks establish the tested instruction-to-plan behavior, not audio
quality or universal natural-language accuracy. No recording was accessed.

Protocol references: [Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs)
and [local chat API](https://docs.ollama.com/api/chat), checked 19 November 2026.
