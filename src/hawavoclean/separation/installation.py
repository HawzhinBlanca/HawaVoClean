"""Read-only optional-runtime and gated-model diagnostics.

This command never accepts license terms, writes credentials, installs packages
or downloads a model. It prints access status, never credential values.
"""

from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import asdict
from typing import Any

from hawavoclean.separation.roformer import probe_roformer_runtime
from hawavoclean.separation.runtime import AUDIOGHOST_COMMIT, probe_runtime


def model_access_status(model_size: str) -> str:
    """Check gated config metadata only when a credential already exists."""
    if model_size not in {"small", "base", "large"}:
        raise ValueError("Unsupported SAM-Audio model")
    try:
        hub = importlib.import_module("huggingface_hub")
    except ImportError:
        return "huggingface_client_not_installed"
    token = hub.get_token()
    if not token:
        return "credential_not_configured"
    try:
        url = hub.hf_hub_url(f"facebook/sam-audio-{model_size}", "config.json")
        hub.get_hf_file_metadata(url, token=token, timeout=10)
        return "access_granted"
    except Exception as exc:
        if type(exc).__name__ == "GatedRepoError":
            return "model_access_not_approved"
        response = getattr(exc, "response", None)
        if getattr(response, "status_code", None) in {401, 403}:
            return "credential_or_model_access_refused"
        return "access_check_unavailable"


def check_installation(
    model_size: str = "small", device: str = "auto", *, check_access: bool = False
) -> dict[str, Any]:
    status = probe_runtime(model_size, device)
    roformer_status = probe_roformer_runtime()
    return {
        "runtime": asdict(status),
        "roformer": asdict(roformer_status),
        "model_access": model_access_status(model_size) if check_access else "not_checked",
        "model_access_page": f"https://huggingface.co/facebook/sam-audio-{model_size}",
        "required_engine_commit": AUDIOGHOST_COMMIT,
        "setup_document": "docs/audioghost-runtime.md",
        "processing_network": "offline_only",
        "inference_quality_verified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-size", choices=("small", "base", "large"), default="small")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument(
        "--check-access",
        action="store_true",
        help="Check gated model metadata online using an existing Hugging Face credential",
    )
    args = parser.parse_args()
    result = check_installation(args.model_size, args.device, check_access=args.check_access)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["runtime"]["available"] else 2)


if __name__ == "__main__":
    main()
