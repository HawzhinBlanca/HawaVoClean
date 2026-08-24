"""Complete, obviously synthetic provenance for reports built by unit tests."""

from hawavoclean.report.schema import (
    BuildMetadata,
    CoreMetadata,
    EnvironmentMetadata,
    GuardMetadata,
    current_build_metadata,
)

TEST_RUNTIME_VERSIONS = {
    "hawavoclean": "test",
    "numpy": "test",
    "scipy": "test",
    "soundfile": "test",
    "pyloudnorm": "test",
    "pydantic": "test",
    "libsndfile": "test",
    "ffmpeg": "test",
    "ffprobe": "test",
}

TEST_DETERMINISTIC_SETTINGS: dict[str, str | int | bool] = {
    "compute_device": "cpu",
    "requested_device": "cpu",
    "worker_pool_size": 1,
    "threads_per_worker": 1,
    "omp_num_threads": "test",
    "mkl_num_threads": "test",
    "python_hash_seed": "test",
    "output_bit_depth": "pcm24",
    "tpdf_dither": True,
    "dither_seed_derivation": "test",
    "result_order": "unit_index",
    "torch_deterministic_algorithms": "test",
}


def build() -> BuildMetadata:
    return current_build_metadata()


def core(core_id: str, algorithm: str, params_hash: str) -> CoreMetadata:
    return CoreMetadata(
        id=core_id,
        algorithm=algorithm,
        params_hash=params_hash,
        lock_sha256="a" * 64,
    )


def guard(guard_id: str, probe_hash: str, calibration_id: str) -> GuardMetadata:
    return GuardMetadata(
        id=guard_id,
        probe_hash=probe_hash,
        calibration_id=calibration_id,
        calibration_sha256="b" * 64,
    )


def environment(
    *,
    platform: str = "test",
    os_version: str = "test",
    python_version: str = "3",
    numpy_version: str = "test",
    scipy_version: str = "test",
    soundfile_version: str = "test",
) -> EnvironmentMetadata:
    return EnvironmentMetadata(
        platform=platform,
        os_version=os_version,
        python_version=python_version,
        numpy_version=numpy_version,
        scipy_version=scipy_version,
        soundfile_version=soundfile_version,
        compute_device="cpu",
        runtime_versions=TEST_RUNTIME_VERSIONS,
        deterministic_settings=TEST_DETERMINISTIC_SETTINGS,
    )
