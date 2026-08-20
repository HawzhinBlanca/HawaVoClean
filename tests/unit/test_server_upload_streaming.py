"""``POST /api/upload`` streams to disk and refuses what would fill it.

Goal box E2: a >= 1 GB upload must not buffer in memory. What Starlette
actually does (verified below, not assumed) is spool each *file* part into a
``SpooledTemporaryFile(max_size=1 MiB)`` — so anything past a megabyte is
already on disk before the route runs, and the route's job is to copy it out
without ever materialising it. That copy is a chunked loop; these tests prove
it stays one, by counting the reads.

Measured on a live engine with a 1.07 GB file: peak RSS 133.8 MB against an
idle 127.8 MB — 6.0 MB of growth for a gigabyte of upload.

The second half is the cap. Without one, a client can write the disk full
twice over (spool + destination). The cap is enforced from ``Content-Length``
before the body is read at all, and again from the byte count while streaming
for a client that declares no length.
"""

from collections.abc import Iterator
from pathlib import Path
from tempfile import SpooledTemporaryFile
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartParser

from hawavoclean.server import app as app_module
from hawavoclean.server.app import (
    DEFAULT_MAX_UPLOAD_BYTES,
    MAX_UPLOAD_BYTES_ENV,
    UPLOAD_CHUNK_BYTES,
    configured_max_upload_bytes,
    create_app,
)
from hawavoclean.server.jobs import JobManager

pytestmark = pytest.mark.unit

TOKEN = "t0ken"
H = {"X-Hawa-Token": TOKEN}


@pytest.fixture
def work(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HAWAVOCLEAN_WORK_DIR", str(tmp_path))
    return tmp_path


def _client(work: Path, **kwargs: Any) -> Iterator[TestClient]:
    assert work.is_dir()
    manager = JobManager()
    app = create_app(TOKEN, None, job_manager=manager, on_shutdown=lambda: None, **kwargs)
    with TestClient(app) as c:
        yield c
    manager.shutdown()


@pytest.fixture
def client(work: Path) -> Iterator[TestClient]:
    yield from _client(work)


def _uploads(work: Path) -> list[Path]:
    root = work / "uploads"
    return sorted(p for p in root.rglob("*") if p.is_file()) if root.is_dir() else []


# --------------------------------------------- what Starlette actually does


def test_starlette_spools_a_file_part_to_disk_above_one_mebibyte() -> None:
    """The engine's memory safety rests on this, so it is asserted rather than
    believed: the multipart parser gives each file part a spooled temp file
    that rolls onto disk the moment it exceeds 1 MiB. ``max_part_size`` looks
    like a cap but applies only to non-file fields, which is why the upload
    route needs its own limit."""
    assert MultiPartParser.spool_max_size == 1024 * 1024
    with SpooledTemporaryFile(max_size=MultiPartParser.spool_max_size) as spool:
        spool.write(b"x" * MultiPartParser.spool_max_size)
        assert spool._rolled is False  # noqa: SLF001 - the behaviour under test
        spool.write(b"x")
        assert spool._rolled is True  # noqa: SLF001


def test_the_body_reaches_the_route_on_disk_not_in_memory(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end through the real request path: by the time the route reads
    the part, ``UploadFile`` reports it is no longer in memory."""
    in_memory: list[bool] = []
    original = UploadFile.read

    async def spy(self: UploadFile, size: int = -1) -> bytes:
        in_memory.append(self._in_memory)  # noqa: SLF001 - the behaviour under test
        return await original(self, size)

    monkeypatch.setattr(UploadFile, "read", spy)
    payload = b"\x11" * (3 * 1024 * 1024)
    r = client.post("/api/upload", headers=H, files={"file": ("big.wav", payload)})
    assert r.status_code == 200, r.text
    assert in_memory and not any(in_memory), "a 3 MiB part must not still be in memory"
    assert Path(r.json()["path"]).read_bytes() == payload


# --------------------------------------------------- the copy loop is chunked


def test_the_write_path_is_chunked(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Shrink the chunk and count the reads. One read of the whole part would
    be a single 5 MiB ``bytes`` object in RAM; the loop must ask for exactly
    the configured chunk each time, as many times as it takes."""
    chunk = 64 * 1024
    monkeypatch.setattr(app_module, "UPLOAD_CHUNK_BYTES", chunk)
    sizes: list[int] = []
    original = UploadFile.read

    async def spy(self: UploadFile, size: int = -1) -> bytes:
        sizes.append(size)
        data = await original(self, size)
        assert len(data) <= chunk, "a read returned more than the chunk size"
        return data

    monkeypatch.setattr(UploadFile, "read", spy)
    payload = bytes(range(256)) * (5 * 1024 * 1024 // 256)  # 5 MiB, not compressible to luck
    r = client.post("/api/upload", headers=H, files={"file": ("chunky.wav", payload)})
    assert r.status_code == 200, r.text

    assert sizes, "the route never read the upload"
    assert set(sizes) == {chunk}, f"the loop asked for something other than {chunk}: {set(sizes)}"
    # len(payload)/chunk full reads, plus the empty read that ends the loop.
    assert len(sizes) == len(payload) // chunk + 1
    saved = Path(r.json()["path"])
    assert saved.read_bytes() == payload
    assert saved.stat().st_size == len(payload)


def test_the_default_chunk_is_a_mebibyte() -> None:
    assert UPLOAD_CHUNK_BYTES == 1024 * 1024


def test_a_failure_part_way_through_leaves_nothing_behind(
    client: TestClient, work: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A half-written file under ``uploads/`` would be indistinguishable from a
    good one to every other endpoint, so the route must clean up after itself."""
    monkeypatch.setattr(app_module, "UPLOAD_CHUNK_BYTES", 4096)
    calls = {"n": 0}
    original = UploadFile.read

    async def flaky(self: UploadFile, size: int = -1) -> bytes:
        calls["n"] += 1
        if calls["n"] == 3:
            raise OSError("disk went away")
        return await original(self, size)

    monkeypatch.setattr(UploadFile, "read", flaky)
    # TestClient re-raises server-side exceptions, which is exactly what makes
    # this a clean assertion: the failure is real, and the tree is still clean.
    with pytest.raises(OSError, match="disk went away"):
        client.post("/api/upload", headers=H, files={"file": ("doomed.wav", b"z" * 65536)})
    assert calls["n"] == 3, "the failure must land part way through the copy"
    assert _uploads(work) == [], "a partial upload was left on disk"


# ------------------------------------------------------------------ the cap


def test_an_oversized_upload_is_refused_with_413(work: Path) -> None:
    """Refused from the declared ``Content-Length``, so the body is never read
    and nothing is written — that is the whole point of the cap."""
    for c in _client(work, max_upload_bytes=4096):
        r = c.post("/api/upload", headers=H, files={"file": ("huge.wav", b"x" * 9000)})
        assert r.status_code == 413, r.text
        body = r.json()
        assert body["error"] == "payload_too_large"
        assert "4096" in body["message"]
        assert _uploads(work) == []


def test_an_upload_inside_the_cap_is_still_accepted(work: Path) -> None:
    for c in _client(work, max_upload_bytes=1024 * 1024):
        payload = b"y" * 4096
        r = c.post("/api/upload", headers=H, files={"file": ("fine.wav", payload)})
        assert r.status_code == 200, r.text
        assert Path(r.json()["path"]).read_bytes() == payload


def test_a_length_less_upload_is_cut_off_while_streaming(work: Path) -> None:
    """A chunked request declares no length, so the ``Content-Length`` check
    cannot fire; the byte counter on the receive channel must stop it instead,
    before the spool file has grown past the cap."""

    def body() -> Iterator[bytes]:
        head = (
            b'------x\r\nContent-Disposition: form-data; name="file"; '
            b'filename="stream.wav"\r\nContent-Type: application/octet-stream\r\n\r\n'
        )
        yield head
        for _ in range(64):
            yield b"\0" * 65536
        yield b"\r\n------x--\r\n"

    for c in _client(work, max_upload_bytes=128 * 1024):
        r = c.post(
            "/api/upload",
            headers={**H, "Content-Type": "multipart/form-data; boundary=----x"},
            content=body(),
        )
        assert r.status_code == 413, r.text
        assert r.json()["error"] == "payload_too_large"
        assert _uploads(work) == []


def test_the_cap_only_guards_the_upload_route(work: Path) -> None:
    """A tiny cap must not start rejecting ordinary API traffic."""
    for c in _client(work, max_upload_bytes=8):
        r = c.get("/api/health", headers=H)
        assert r.status_code == 200
        r = c.post("/api/analyze", headers=H, json={"path": str(work / "nope.wav")})
        assert r.status_code == 404  # missing file, not 413


def test_an_unauthenticated_oversized_upload_is_401_not_413(work: Path) -> None:
    """The token check is outside the cap, so a flood from an unauthenticated
    client is rejected on the cheaper test first."""
    for c in _client(work, max_upload_bytes=16):
        r = c.post("/api/upload", files={"file": ("huge.wav", b"x" * 9000)})
        assert r.status_code == 401
        assert r.json()["error"] == "unauthorized"


def test_a_cap_of_zero_disables_the_limit(work: Path) -> None:
    for c in _client(work, max_upload_bytes=0):
        r = c.post("/api/upload", headers=H, files={"file": ("free.wav", b"x" * 200000)})
        assert r.status_code == 200, r.text


# ------------------------------------------------------------ configuration


def test_the_cap_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(MAX_UPLOAD_BYTES_ENV, raising=False)
    assert configured_max_upload_bytes() == DEFAULT_MAX_UPLOAD_BYTES
    assert DEFAULT_MAX_UPLOAD_BYTES > 2 * 1024**3, "a 3-hour stereo WAV must fit by default"

    monkeypatch.setenv(MAX_UPLOAD_BYTES_ENV, "123456")
    assert configured_max_upload_bytes() == 123456
    monkeypatch.setenv(MAX_UPLOAD_BYTES_ENV, "0")
    assert configured_max_upload_bytes() == 0
    # A typo must not silently uncap the endpoint.
    for junk in ("", "   ", "lots", "-5", "1.5"):
        monkeypatch.setenv(MAX_UPLOAD_BYTES_ENV, junk)
        assert configured_max_upload_bytes() == DEFAULT_MAX_UPLOAD_BYTES, junk


def test_the_environment_cap_reaches_a_built_app(
    work: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(MAX_UPLOAD_BYTES_ENV, "4096")
    for c in _client(work):
        assert c.app.state.max_upload_bytes == 4096  # type: ignore[attr-defined]
        r = c.post("/api/upload", headers=H, files={"file": ("huge.wav", b"x" * 9000)})
        assert r.status_code == 413
