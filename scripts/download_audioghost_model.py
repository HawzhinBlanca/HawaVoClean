"""Explicit post-approval model setup; never accepts terms or performs inference."""

import argparse
import importlib
import json
import os
from pathlib import Path


def download(model_size: str, check: bool) -> int:
    hf_hub = importlib.import_module("huggingface_hub")
    hf_api_cls = hf_hub.HfApi
    get_token = hf_hub.get_token
    snapshot_download = hf_hub.snapshot_download

    token = get_token()
    if not token:
        print("No Hugging Face credential is configured; request SAM-Audio model access first.")
        print(
            "Log in locally: .local-ai/audioghost-env/bin/hf auth login (never paste tokens into chat)."
        )
        return 2
    if check:
        print("Credential configured. No files downloaded or model terms accepted.")
        return 0
    try:
        api = hf_api_cls(token=token)
        model = f"facebook/sam-audio-{model_size}"
        directory = Path(
            snapshot_download(
                model,
                token=token,
                allow_patterns=["config.json", "checkpoint.pt"],
            )
        )
        config = json.loads((directory / "config.json").read_text())
        encoder = (config.get("text_encoder") or {}).get("name", "t5-base")
        text_info = api.model_info(encoder)
        files = {item.rfilename for item in text_info.siblings or []}
        if "model.safetensors" in files:
            weights = ["model.safetensors"]
        elif "model.safetensors.index.json" in files:
            weights = ["model.safetensors.index.json", "model-*.safetensors"]
        elif "pytorch_model.bin" in files:
            weights = ["pytorch_model.bin"]
        else:
            raise ValueError("Unsupported text-encoder weight layout")
        snapshot_download(
            encoder,
            token=token,
            allow_patterns=[
                "config.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "spiece.model",
                "tokenizer.json",
                "added_tokens.json",
                *weights,
            ],
        )
        print(f"Downloaded {model} at {directory.name} and its {encoder} text encoder.")
        print("No model terms accepted or inference performed by this command.")
        return 0
    except Exception as error:
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None)
        if type(error).__name__ == "GatedRepoError" or status in {401, 403}:
            print("Model access refused; request approval and check the local Hugging Face login.")
            print("https://huggingface.co/facebook/sam-audio-" + model_size)
        else:
            print("Model setup failed (" + type(error).__name__ + "). No inference was performed.")
        return 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-size", choices=("small", "base", "large"), default="large")
    parser.add_argument("--check", action="store_true", help="Check login without downloading")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.worker:
        from hawavoclean.separation.runtime import SeparationError, _runtime_paths

        try:
            python, _root = _runtime_paths()
        except SeparationError as error:
            print(str(error))
            return 2
        command = [str(python), "-I", "-B", str(Path(__file__).resolve()), "--worker"]
        command.extend(["--model-size", args.model_size])
        if args.check:
            command.append("--check")
        os.execv(str(python), command)
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ.pop(name, None)
    return download(args.model_size, args.check)


if __name__ == "__main__":
    raise SystemExit(main())
