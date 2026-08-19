"""Every voiceclean command the README documents must actually exist.

Parses fenced bash blocks out of README.md, extracts voiceclean
subcommands, and asserts each is registered in the CLI — so command drift
between docs and code is a test failure, not a support ticket.
"""

import re
import subprocess
import sys
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
    proc = subprocess.run(
        [sys.executable, "-m", "voiceclean.cli", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    m = re.search(r"\{([a-z0-9,-]+)\}", proc.stdout)
    assert m, f"could not parse subcommands from --help output:\n{proc.stdout}"
    return set(m.group(1).split(","))


def test_every_documented_command_is_registered() -> None:
    documented = _documented_subcommands()
    assert documented, "README documents no voiceclean commands — parsing broke?"
    registered = _registered_subcommands()
    missing = documented - registered
    assert not missing, (
        f"README documents commands that do not exist: {sorted(missing)} "
        f"(registered: {sorted(registered)})"
    )
