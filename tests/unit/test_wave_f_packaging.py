"""Unit tests for Wave F packaging: portable FFmpeg relocation & Windows win32-x64 bundler."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import build_resolve_engine, build_windows_engine


def test_parse_otool_l_and_rpaths() -> None:
    otool_l_output = (
        "/tmp/stage/bin/ffmpeg:\n"
        "\t/opt/homebrew/opt/ffmpeg/lib/libavcodec.61.dylib (compatibility version 61.0.0, current version 61.3.100)\n"
        "\t@rpath/libavutil.59.dylib (compatibility version 59.0.0, current version 59.8.100)\n"
        "\t/System/Library/Frameworks/CoreAudio.framework/Versions/A/CoreAudio (compatibility version 1.0.0)\n"
        "\t/usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1351.0.0)\n"
    )
    deps = build_resolve_engine.parse_otool_l(otool_l_output)
    assert deps == [
        "/opt/homebrew/opt/ffmpeg/lib/libavcodec.61.dylib",
        "@rpath/libavutil.59.dylib",
        "/System/Library/Frameworks/CoreAudio.framework/Versions/A/CoreAudio",
        "/usr/lib/libSystem.B.dylib",
    ]
    assert build_resolve_engine.is_forbidden_dylib(deps[0]) is True
    assert build_resolve_engine.is_non_system_dylib(deps[0]) is True
    assert build_resolve_engine.is_non_system_dylib(deps[1]) is True
    assert build_resolve_engine.is_system_dylib(deps[2]) is True
    assert build_resolve_engine.is_system_dylib(deps[3]) is True

    otool_l_load = (
        "Load command 24\n"
        "          cmd LC_RPATH\n"
        "      cmdsize 40\n"
        "         path @loader_path/../lib (offset 12)\n"
        "Load command 25\n"
        "          cmd LC_RPATH\n"
        "      cmdsize 48\n"
        "         path /opt/homebrew/lib (offset 12)\n"
    )
    assert build_resolve_engine.parse_otool_rpaths(otool_l_load) == [
        "@loader_path/../lib",
        "/opt/homebrew/lib",
    ]


def test_relocate_or_verify_ffmpeg_binary_relocates_transitive_dylibs_and_signs(
    tmp_path: Path,
) -> None:
    # Create fake source binary and two transitive dylibs (libavcodec -> libavutil)
    fake_prefix = tmp_path / "homebrew"
    src_bin_dir = fake_prefix / "bin"
    src_lib_dir = fake_prefix / "lib"
    src_bin_dir.mkdir(parents=True)
    src_lib_dir.mkdir(parents=True)

    src_ffmpeg = src_bin_dir / "ffmpeg"
    src_avcodec = src_lib_dir / "libavcodec.61.dylib"
    src_avutil = src_lib_dir / "libavutil.59.dylib"
    src_ffmpeg.write_bytes(b"\xcf\xfa\xed\xfe-ffmpeg")
    src_avcodec.write_bytes(b"\xcf\xfa\xed\xfe-avcodec")
    src_avutil.write_bytes(b"\xcf\xfa\xed\xfe-avutil")
    # Simulate read-only Homebrew dylib permissions
    src_avcodec.chmod(0o444)
    src_avutil.chmod(0o444)

    stage = tmp_path / "stage"
    dst_ffmpeg = stage / "bin" / "ffmpeg"
    dst_lib_dir = stage / "lib"

    # Simulated Mach-O dependency table mutated in-memory by fake install_name_tool
    deps_by_path: dict[str, list[str]] = {
        str(dst_ffmpeg): [
            str(src_avcodec),
            "/usr/lib/libSystem.B.dylib",
        ],
        str(dst_lib_dir / "libavcodec.61.dylib"): [
            str(src_avcodec),  # self LC_ID_DYLIB
            "@rpath/libavutil.59.dylib",
            "/usr/lib/libSystem.B.dylib",
        ],
        str(dst_lib_dir / "libavutil.59.dylib"): [
            str(src_avutil),  # self LC_ID_DYLIB
            "/usr/lib/libSystem.B.dylib",
        ],
    }
    recorded_commands: list[list[str]] = []

    def fake_runner(cmd: list[str]) -> str:
        recorded_commands.append(list(cmd))
        if cmd[:2] == ["otool", "-L"]:
            target = cmd[2]
            entries = deps_by_path[target]
            return f"{target}:\n" + "".join(
                f"\t{dep} (compatibility version 1.0.0, current version 1.0.0)\n" for dep in entries
            )
        if cmd[:2] == ["otool", "-l"]:
            return f"          cmd LC_RPATH\n         path {src_lib_dir} (offset 12)\n"
        if cmd[:2] == ["install_name_tool", "-id"]:
            new_id, target = cmd[2], cmd[3]
            deps_by_path[target][0] = new_id
            return ""
        if cmd[:2] == ["install_name_tool", "-change"]:
            old_ref, new_ref, target = cmd[2], cmd[3], cmd[4]
            deps_by_path[target] = [new_ref if d == old_ref else d for d in deps_by_path[target]]
            return ""
        if cmd[:3] == ["codesign", "--force", "--sign"]:
            return ""
        raise AssertionError(f"unexpected command: {cmd}")

    relocated = build_resolve_engine.relocate_or_verify_ffmpeg_binary(
        src_ffmpeg,
        dst_ffmpeg,
        dst_lib_dir,
        run_cmd=fake_runner,
        is_macos=True,
    )

    assert dst_ffmpeg.is_file()
    assert [p.name for p in relocated] == ["libavcodec.61.dylib", "libavutil.59.dylib"]
    assert (dst_lib_dir / "libavcodec.61.dylib").is_file()
    assert (dst_lib_dir / "libavutil.59.dylib").is_file()

    # Verify rewritten references
    assert deps_by_path[str(dst_ffmpeg)] == [
        "@loader_path/../lib/libavcodec.61.dylib",
        "/usr/lib/libSystem.B.dylib",
    ]
    assert deps_by_path[str(dst_lib_dir / "libavcodec.61.dylib")] == [
        "@loader_path/libavcodec.61.dylib",
        "@loader_path/libavutil.59.dylib",
        "/usr/lib/libSystem.B.dylib",
    ]
    assert deps_by_path[str(dst_lib_dir / "libavutil.59.dylib")] == [
        "@loader_path/libavutil.59.dylib",
        "/usr/lib/libSystem.B.dylib",
    ]

    # Verify ad-hoc codesigning ran for ffmpeg and both dylibs
    signed_targets = {
        cmd[4] for cmd in recorded_commands if cmd[:4] == ["codesign", "--force", "--sign", "-"]
    }
    assert signed_targets == {
        str(dst_ffmpeg),
        str(dst_lib_dir / "libavcodec.61.dylib"),
        str(dst_lib_dir / "libavutil.59.dylib"),
    }


def test_verify_macho_rejects_remaining_homebrew_and_usr_local_references(tmp_path: Path) -> None:
    fake_bin = tmp_path / "ffmpeg"
    fake_bin.write_bytes(b"macho")

    for forbidden in (
        "/opt/homebrew/opt/x264/lib/libx264.164.dylib",
        "/usr/local/opt/lame/lib/libmp3lame.0.dylib",
        "@rpath/libunrelocated.dylib",
    ):

        def fake_otool(_cmd: list[str], bad_dep: str = forbidden) -> str:
            return (
                f"{fake_bin}:\n"
                f"\t{bad_dep} (compatibility version 1.0.0, current version 1.0.0)\n"
                "\t/usr/lib/libSystem.B.dylib (compatibility version 1.0.0)\n"
            )

        with pytest.raises(
            build_resolve_engine.BundleError, match="portability verification failed"
        ):
            build_resolve_engine.verify_macho_no_forbidden_dylibs([fake_bin], run_cmd=fake_otool)


def test_relocate_fails_loudly_if_homebrew_ref_survives_rewrite(tmp_path: Path) -> None:
    src_bin = tmp_path / "ffmpeg"
    src_bin.write_bytes(b"macho")
    dst_bin = tmp_path / "stage" / "bin" / "ffmpeg"
    lib_dir = tmp_path / "stage" / "lib"

    # Simulate a broken install_name_tool that leaves /opt/homebrew/... untouched
    fake_lib = tmp_path / "libstub.dylib"
    fake_lib.write_bytes(b"dylib")

    def stubborn_runner(cmd: list[str]) -> str:
        if cmd[:2] == ["otool", "-L"]:
            if cmd[2] == str(dst_bin):
                return (
                    f"{dst_bin}:\n\t/opt/homebrew/lib/libstub.dylib (compatibility version 1.0.0)\n"
                )
            return f"{cmd[2]}:\n\t@loader_path/libstub.dylib (compatibility version 1.0.0)\n"
        return ""

    # Unresolvable /opt/homebrew path must fail loudly
    with pytest.raises(build_resolve_engine.BundleError, match="Cannot resolve non-system dylib"):
        build_resolve_engine.relocate_or_verify_ffmpeg_binary(
            src_bin,
            dst_bin,
            lib_dir,
            run_cmd=stubborn_runner,
            is_macos=True,
        )


def test_windows_bundler_manifest_checksums_and_verification(tmp_path: Path) -> None:
    identity = build_windows_engine.load_release_identity()
    version = identity["version"]
    assert version == "3.3.0"

    # Stage fake PE binaries for ffmpeg.exe, ffprobe.exe, an adjacent DLL, and launcher stub
    ffmpeg_src = tmp_path / "ffmpeg-win"
    ffmpeg_src.mkdir()
    (ffmpeg_src / "ffmpeg.exe").write_bytes(b"MZ-fake-ffmpeg-pe-payload")
    (ffmpeg_src / "ffprobe.exe").write_bytes(b"MZ-fake-ffprobe-pe-payload")
    (ffmpeg_src / "avcodec-61.dll").write_bytes(b"MZ-fake-avcodec-dll")

    launcher_stub = tmp_path / "stub.exe"
    launcher_stub.write_bytes(b"MZ-fake-launcher-stub")

    stage = tmp_path / "win32-x64-stage"
    stage.mkdir()

    staged_tools = build_windows_engine.stage_windows_ffmpeg(
        stage / "bin",
        ffmpeg_dir=ffmpeg_src,
    )
    assert staged_tools == ["ffmpeg.exe", "ffprobe.exe"]
    assert (stage / "bin" / "ffmpeg.exe").is_file()
    assert (stage / "bin" / "ffprobe.exe").is_file()
    assert (stage / "bin" / "avcodec-61.dll").is_file()

    build_windows_engine.write_windows_launchers(stage, launcher_stub=launcher_stub)
    assert (stage / "hawavoclean-engine.exe").is_file()
    assert (stage / "hawavoclean-engine.cmd").is_file()
    assert (stage / "launcher.py").is_file()

    manifest = build_windows_engine.build_manifest_dict(
        version=version,
        product=identity["product"],
        source_revision="a" * 40,
        source_date_epoch=1700000000,
        dependency_lock_sha256="b" * 64,
        wheel_filename=f"hawavoclean-{version}-py3-none-any.whl",
        wheel_sha256="c" * 64,
        python_version="3.11.11",
        installed_distributions=[f"hawavoclean=={version}"],
        bundled_tools=staged_tools,
    )
    build_windows_engine.write_manifests(stage, manifest)

    # Post-stage verification must succeed cleanly
    report = build_windows_engine.verify_staged_bundle(stage, expected_version=version)
    assert report["platform"] == "win32"
    assert report["arch"] == "x64"
    assert report["version"] == version
    assert report["verified_files"] >= 7

    loaded_manifest = json.loads((stage / "ENGINE-MANIFEST.json").read_text(encoding="utf-8"))
    assert loaded_manifest["platform"] == "win32"
    assert loaded_manifest["arch"] == "x64"
    assert loaded_manifest["version"] == version
    assert loaded_manifest["product_version"] == version
    assert loaded_manifest["launcher"] == "hawavoclean-engine.exe"
    assert loaded_manifest["bundled_tools"] == ["ffmpeg.exe", "ffprobe.exe"]

    # 1. Tampering with a file must fail verification
    (stage / "launcher.py").write_text("# tampered\n", encoding="utf-8")
    with pytest.raises(build_windows_engine.WindowsBundleError, match="checksum mismatch"):
        build_windows_engine.verify_staged_bundle(stage, expected_version=version)

    # Restore clean manifests
    build_windows_engine.write_manifests(stage, manifest)
    assert (
        build_windows_engine.verify_staged_bundle(stage, expected_version=version)["verified_files"]
        >= 7
    )

    # 2. Injecting README.txt placeholder must fail verification
    readme = stage / "README.txt"
    readme.write_text("placeholder", encoding="utf-8")
    with pytest.raises(build_windows_engine.WindowsBundleError, match="placeholder"):
        build_windows_engine.verify_staged_bundle(stage, expected_version=version)
    readme.unlink()

    # 3. Path traversal in ENGINE-SHA256SUMS must fail verification
    outside = tmp_path / "escape.txt"
    outside.write_text("escape", encoding="utf-8")
    outside_hash = hashlib.sha256(outside.read_bytes()).hexdigest()
    checksums = stage / "ENGINE-SHA256SUMS"
    checksums.write_text(
        checksums.read_text(encoding="utf-8") + f"{outside_hash}  ./../escape.txt\n",
        encoding="utf-8",
    )
    with pytest.raises(build_windows_engine.WindowsBundleError, match="escapes bundle root"):
        build_windows_engine.verify_staged_bundle(stage, expected_version=version)


def test_windows_ffmpeg_staging_refuses_missing_and_invalid_pe(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty-ffmpeg"
    empty_dir.mkdir()

    # Missing ffmpeg.exe fails unless allow_missing_ffmpeg=True
    with pytest.raises(build_windows_engine.WindowsBundleError, match="ffmpeg.exe"):
        build_windows_engine.stage_windows_ffmpeg(tmp_path / "out1", ffmpeg_dir=empty_dir)

    assert (
        build_windows_engine.stage_windows_ffmpeg(
            tmp_path / "out2", ffmpeg_dir=empty_dir, allow_missing_ffmpeg=True
        )
        == []
    )

    # Invalid non-PE binary fails PE header check
    bad_dir = tmp_path / "bad-ffmpeg"
    bad_dir.mkdir()
    (bad_dir / "ffmpeg.exe").write_bytes(b"#!/bin/sh\nnot-a-pe\n")
    (bad_dir / "ffprobe.exe").write_bytes(b"MZ-valid")
    with pytest.raises(build_windows_engine.WindowsBundleError, match="PE \\(MZ\\) header"):
        build_windows_engine.stage_windows_ffmpeg(tmp_path / "out3", ffmpeg_dir=bad_dir)
