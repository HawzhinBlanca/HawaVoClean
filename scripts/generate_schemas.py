#!/usr/bin/env python3
"""Regenerate the JSON Schema documents from the Pydantic models.

The schemas under docs/schemas/ are generated artifacts. Run this after
changing config.py or report/schema.py, and commit the result.
"""

import json
from pathlib import Path

from hawavoclean.config import HawaVoCleanConfig
from hawavoclean.report.schema import CorpusManifest, HawaVoCleanReport

OUT = Path(__file__).resolve().parents[1] / "docs" / "schemas"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, model in (
        ("config.schema.json", HawaVoCleanConfig),
        ("report.schema.json", HawaVoCleanReport),
        ("corpus.schema.json", CorpusManifest),
    ):
        schema = model.model_json_schema()
        (OUT / name).write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {OUT / name}")


if __name__ == "__main__":
    main()
