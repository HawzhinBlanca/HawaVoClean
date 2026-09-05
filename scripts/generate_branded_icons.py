#!/usr/bin/env python3
"""Generate pixel-perfect HawaVoClean branded visual identity assets.

Renders high-resolution raster and multi-format icon assets (ICNS, ICO, PNG)
conforming to the canonical HawaVoClean brand mark defined in ui/index.html:
five rounded waveform bars transitioning from amber (#ffb347 / #ffd28a) to
cyan (#39d0ff) on a dark squircle (#14171c with #2a3139 border).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DESKTOP_BUILD = ROOT / "desktop" / "build"
DESKTOP_BRANDING = ROOT / "desktop" / "resources" / "branding"

# Color palette from ui/index.html
BG_FILL = "#14171c"
BORDER_COLOR = "#2a3139"
BAR_1_2_COLOR = "#ffb347"
BAR_3_COLOR = "#ffd28a"
BAR_4_5_COLOR = "#39d0ff"


def render_master_icon(size: int = 4096) -> Image.Image:
    """Render a 4096x4096 supersampled RGBA image of the brand mark."""
    master = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(master)
    scale = size / 32.0

    # Background squircle with border
    pad = 0.5 * scale
    radius = 7.0 * scale
    border_width = int(round(1.0 * scale))
    draw.rounded_rectangle(
        [pad, pad, size - pad, size - pad],
        radius=radius,
        fill=BG_FILL,
        outline=BORDER_COLOR,
        width=border_width,
    )

    # 5 rounded waveform bars
    line_w = int(round(3.0 * scale))
    r_cap = line_w / 2.0

    bars = [
        (6.5, 12.0, 20.0, BAR_1_2_COLOR),
        (11.25, 8.0, 24.0, BAR_1_2_COLOR),
        (16.0, 5.0, 27.0, BAR_3_COLOR),
        (20.75, 9.0, 23.0, BAR_4_5_COLOR),
        (25.5, 11.5, 20.5, BAR_4_5_COLOR),
    ]

    for bx, by1, by2, color in bars:
        cx = bx * scale
        y1 = by1 * scale
        y2 = by2 * scale
        # Draw central line
        draw.line([(cx, y1 + r_cap), (cx, y2 - r_cap)], fill=color, width=line_w)
        # Draw smooth round caps
        draw.ellipse([cx - r_cap, y1, cx + r_cap, y1 + 2 * r_cap], fill=color)
        draw.ellipse([cx - r_cap, y2 - 2 * r_cap, cx + r_cap, y2], fill=color)

    return master.resize((1024, 1024), Image.Resampling.LANCZOS)


def generate_all_icons(dest_dirs: list[Path]) -> None:
    """Generate all icon formats and populate destination directories."""
    master_1024 = render_master_icon()

    for dest in dest_dirs:
        dest.mkdir(parents=True, exist_ok=True)
        icons_subdir = dest / "icons"
        icons_subdir.mkdir(exist_ok=True)

        # 1. Master PNG (1024x1024)
        master_png = dest / "icon.png"
        master_1024.save(master_png, "PNG")

        # 2. Standard Linux PNG sizes
        linux_sizes = [16, 32, 48, 64, 128, 256, 512, 1024]
        for s in linux_sizes:
            resized = (
                master_1024 if s == 1024 else master_1024.resize((s, s), Image.Resampling.LANCZOS)
            )
            resized.save(icons_subdir / f"{s}x{s}.png", "PNG")

        # 3. Windows ICO multi-resolution
        ico_path = dest / "icon.ico"
        master_1024.save(
            ico_path,
            format="ICO",
            sizes=[
                (16, 16),
                (24, 24),
                (32, 32),
                (48, 48),
                (64, 64),
                (128, 128),
                (256, 256),
            ],
        )

        # 4. Apple ICNS (using iconutil if on macOS)
        icns_path = dest / "icon.icns"
        with tempfile.TemporaryDirectory() as td:
            iconset = Path(td) / "icon.iconset"
            iconset.mkdir()
            apple_sizes = [
                ("icon_16x16.png", 16),
                ("icon_16x16@2x.png", 32),
                ("icon_32x32.png", 32),
                ("icon_32x32@2x.png", 64),
                ("icon_128x128.png", 128),
                ("icon_128x128@2x.png", 256),
                ("icon_256x256.png", 256),
                ("icon_256x256@2x.png", 512),
                ("icon_512x512.png", 512),
                ("icon_512x512@2x.png", 1024),
            ]
            for name, s in apple_sizes:
                resized = (
                    master_1024
                    if s == 1024
                    else master_1024.resize((s, s), Image.Resampling.LANCZOS)
                )
                resized.save(iconset / name, "PNG")

            if shutil.which("iconutil"):
                subprocess.run(
                    ["iconutil", "-c", "icns", os.fspath(iconset), "-o", os.fspath(icns_path)],
                    check=True,
                    capture_output=True,
                )
            else:
                # Fallback: copy largest PNG as ICNS placeholder if iconutil is missing
                shutil.copyfile(master_png, icns_path)

        print(f"Generated branded assets in {dest.relative_to(ROOT)}:")
        print(f"  - icon.png (1024x1024 master, {master_png.stat().st_size} bytes)")
        print(f"  - icon.icns (macOS multi-res, {icns_path.stat().st_size} bytes)")
        print(f"  - icon.ico (Windows multi-res, {ico_path.stat().st_size} bytes)")
        print(f"  - icons/ (Linux sizes: {', '.join(f'{s}x{s}' for s in linux_sizes)})")


def main() -> int:
    generate_all_icons([DESKTOP_BUILD, DESKTOP_BRANDING])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
