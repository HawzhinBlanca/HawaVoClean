"""Every voiceclean command the README documents must actually exist.

Parses fenced bash blocks out of README.md, extracts voiceclean
subcommands, and asserts each is registered in the argparse CLI — so
command drift between docs and code is a test failure, not a support
ticket.
"""

import re
from pathlib import Path

README = Path(__file__).resolve().parents[2] / "README.md"


def _documented_subcommands() -> set[str]:
    text = README.read_text(encoding="utf-8")
    blocks = re.findall(r"```bash\n(.*?)```", text, flags=re.S)
    subcommands: set[str] = set()
    for block in blocks:
        for line in block.splitlines():
            m = re.match(r"\s*voiceclean\s+([a-z][a-z0-9-]*)", line)
            if m:
                subcommands.add(m.group(1))
    return subcommands


def _registered_subcommands() -> set[str]:
    import argparse

    import voiceclean.cli as cli

    captured: dict[str, set[str]] = {}
    real_add_subparsers = argparse.ArgumentParser.add_subparsers

    def spy(self: argparse.ArgumentParser, **kw: object) -> object:
        action = real_add_subparsers(self, **kw)  # type: ignore[arg-type]
        real_add_parser = action.add_parser

        def add_parser_spy(name: str, **kw2: object) -> argparse.ArgumentParser:
            captured.setdefault("names", set()).add(name)
            return real_add_parser(name, **kw2)  # type: ignore[arg-type]

        action.add_parser = add_parser_spy  # type: ignore[method-assign]
        return action

    argparse.ArgumentParser.add_subparsers = spy  # type: ignore[method-assign]
    try:
        try:
            import sys

            argv = sys.argv
            sys.argv = ["voiceclean", "--version"]
            cli.main()
        except SystemExit:
            pass
        finally:
            sys.argv = argv
    finally:
        argparse.ArgumentParser.add_subparsers = real_add_subparsers  # type: ignore[method-assign]
    return captured.get("names", set())


def test_every_documented_command_is_registered() -> None:
    documented = _documented_subcommands()
    assert documented, "README documents no voiceclean commands — parsing broke?"
    registered = _registered_subcommands()
    missing = documented - registered
    assert not missing, (
        f"README documents commands that do not exist: {sorted(missing)} "
        f"(registered: {sorted(registered)})"
    )
