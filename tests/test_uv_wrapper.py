"""Behavioral pins for the host-local, fail-loud uv cache wrapper."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WRAPPER = REPO / "scripts" / "uv.sh"


def _fake_uv(bin_dir: Path) -> None:
    command = bin_dir / "uv"
    command.write_text("#!/usr/bin/env bash\nprintf '%s\\n' \"$UV_CACHE_DIR\"\n", encoding="utf-8")
    command.chmod(command.stat().st_mode | stat.S_IXUSR)


def test_wrapper_falls_back_outside_repo_when_home_cache_is_unusable(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _fake_uv(bin_dir)
    blocked_home = tmp_path / "not-a-directory"
    blocked_home.write_text("blocked", encoding="utf-8")
    temp_root = tmp_path / "tmp"
    environment = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(blocked_home),
        "TMPDIR": str(temp_root),
    }
    environment.pop("XDG_CACHE_HOME", None)
    environment.pop("UV_CACHE_DIR", None)
    result = subprocess.run(
        ["bash", str(WRAPPER), "--version"],
        capture_output=True,
        check=True,
        text=True,
        env=environment,
    )
    selected_cache = Path(result.stdout.strip())
    assert selected_cache.name == "broadway-uv-cache"
    assert REPO not in selected_cache.parents
    assert "UV CACHE NOTICE" in result.stderr


def test_wrapper_fails_loudly_for_an_explicit_unwritable_cache(tmp_path: Path) -> None:
    blocked_cache = tmp_path / "not-a-directory"
    blocked_cache.write_text("blocked", encoding="utf-8")
    result = subprocess.run(
        ["bash", str(WRAPPER), "--version"],
        capture_output=True,
        check=False,
        text=True,
        env={**os.environ, "UV_CACHE_DIR": str(blocked_cache)},
    )
    assert result.returncode == 1
    assert "UV CACHE ERROR: requested UV_CACHE_DIR is not writable" in result.stderr


def test_wrapper_fails_loudly_when_no_default_or_fallback_cache_is_usable(tmp_path: Path) -> None:
    blocked_home = tmp_path / "blocked-home"
    blocked_home.write_text("blocked", encoding="utf-8")
    blocked_temp = tmp_path / "blocked-temp"
    blocked_temp.write_text("blocked", encoding="utf-8")
    environment = {**os.environ, "HOME": str(blocked_home), "TMPDIR": str(blocked_temp)}
    environment.pop("XDG_CACHE_HOME", None)
    environment.pop("UV_CACHE_DIR", None)
    result = subprocess.run(
        ["bash", str(WRAPPER), "--version"],
        capture_output=True,
        check=False,
        text=True,
        env=environment,
    )
    assert result.returncode == 1
    assert "UV CACHE ERROR: neither preferred cache" in result.stderr
