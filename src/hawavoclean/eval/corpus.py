"""Corpus manifest loading, schema validation, and split integrity verification."""

import json
from pathlib import Path

from hawavoclean.errors import InvalidUserInputError
from hawavoclean.hashing import hash_file, hash_json_canonical
from hawavoclean.report.schema import CorpusItem, CorpusManifest


def load_corpus_manifest(manifest_path: Path | str) -> CorpusManifest:
    """Load and validate corpus manifest from JSON or JSONL file."""
    p = Path(manifest_path).resolve()
    if not p.exists():
        raise InvalidUserInputError(f"Manifest file not found: {p}")

    items: list[CorpusItem] = []
    with open(p, encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "{":
            # Either JSON manifest or single-line JSONL
            try:
                data = json.load(f)
                if "items" in data:
                    return CorpusManifest.model_validate(data)
                else:
                    # Single item JSON
                    items.append(CorpusItem.model_validate(data))
            except Exception:
                f.seek(0)
                for line in f:
                    line = line.strip()
                    if line:
                        items.append(CorpusItem.model_validate(json.loads(line)))
        else:
            for line in f:
                line = line.strip()
                if line:
                    items.append(CorpusItem.model_validate(json.loads(line)))

    # Deterministic sorting by item ID
    items = sorted(items, key=lambda x: x.id)
    manifest_id = p.stem
    split_name = items[0].split if items else "unknown"

    canonical_dict = [it.model_dump(mode="json") for it in items]
    manifest_sha = hash_json_canonical(canonical_dict)

    return CorpusManifest(
        schema_version=1,
        manifest_id=manifest_id,
        split_name=split_name,
        items_count=len(items),
        manifest_sha256=manifest_sha,
        items=items,
    )


def verify_corpus_audio_files(manifest: CorpusManifest, base_dir: Path | None = None) -> None:
    """Verify that every audio file exists and matches its expected SHA-256 digest."""
    for item in manifest.items:
        audio_path = Path(item.audio_path)
        if not audio_path.is_absolute() and base_dir is not None:
            audio_path = base_dir / audio_path

        if not audio_path.exists():
            raise InvalidUserInputError(f"Corpus item {item.id} audio missing: {audio_path}")

        actual_sha = hash_file(audio_path)
        if item.audio_sha256 and actual_sha != item.audio_sha256:
            raise InvalidUserInputError(
                f"Corpus item {item.id} SHA-256 mismatch: expected {item.audio_sha256}, got {actual_sha}"
            )
