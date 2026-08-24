#!/usr/bin/env python3
"""Regenerate or verify the JSON Schema documents from the Pydantic models.

The schemas under docs/schemas/ are generated artifacts. Run this after
changing config.py or report/schema.py, and commit the result. Release checks
use ``--check`` so verification never rewrites the checkout.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from hawavoclean.config import HawaVoCleanConfig
from hawavoclean.report.schema import CorpusManifest, HawaVoCleanReport

OUT = Path(__file__).resolve().parents[1] / "docs" / "schemas"
MODELS = (
    ("config.schema.json", HawaVoCleanConfig),
    ("report.schema.json", HawaVoCleanReport),
    ("corpus.schema.json", CorpusManifest),
)


def rendered_schemas() -> dict[str, str]:
    """Return the canonical generated documents without touching the filesystem."""
    rendered: dict[str, str] = {}
    for name, model in MODELS:
        schema: dict[str, Any] = model.model_json_schema()
        rendered[name] = json.dumps(schema, indent=2) + "\n"
    return rendered


def check_schemas(output_dir: Path = OUT) -> list[str]:
    """Return drift descriptions; an empty result means the tree is current."""
    expected = rendered_schemas()
    drift: list[str] = []
    for name, content in expected.items():
        path = output_dir / name
        if not path.is_file():
            drift.append(f"missing generated schema: {path}")
        elif path.read_text(encoding="utf-8") != content:
            drift.append(f"generated schema differs: {path}")
    if output_dir.is_dir():
        for path in sorted(output_dir.glob("*.schema.json")):
            if path.name not in expected:
                drift.append(f"unexpected generated schema: {path}")
    return drift


def write_schemas(output_dir: Path = OUT) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in rendered_schemas().items():
        path = output_dir / name
        path.write_text(content, encoding="utf-8")
        print(f"wrote {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail on missing, stale or unexpected schema files without writing",
    )
    args = parser.parse_args()
    if args.check:
        drift = check_schemas()
        if drift:
            for message in drift:
                print(message)
            return 1
        print(f"generated schemas are current: {OUT}")
        return 0
    write_schemas()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
