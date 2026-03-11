from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from filter_utils import load_config, normalized_base_url


PATCH_REF = "14604"
PATCH_URL = f"https://patch-diff.githubusercontent.com/raw/ollama/ollama/pull/{PATCH_REF}.patch"
PATCH_PORT = int(os.getenv("ZIKA_OLLAMA_PATCH_PORT", "11437"))
CACHE_ROOT = Path.home() / ".cache" / "zika" / f"ollama-pr{PATCH_REF}"
SRC_DIR = CACHE_ROOT / "src"
PATCH_FILE = CACHE_ROOT / f"pr{PATCH_REF}.patch"
BIN_PATH = CACHE_ROOT / "ollama"
BUILD_LOG = REPO_ROOT / "zika" / "logs" / f"ollama_pr{PATCH_REF}_build.log"
SERVE_LOG = REPO_ROOT / "zika" / "logs" / f"ollama_pr{PATCH_REF}_serve.log"
CONFIG_PATH = REPO_ROOT / "zika" / "config" / "zika.yaml"
STAGE2_SCRIPT = CURRENT_DIR / "02_filter_headlines_ensemble.py"


def setup_logger() -> logging.Logger:
    BUILD_LOG.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("zika_stage2_runner")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def append_to_log(log_path: Path, text: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(text)


def run_and_log(cmd: list[str], cwd: Path, log_path: Path) -> None:
    append_to_log(log_path, f"\n$ {' '.join(cmd)}\n")
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    append_to_log(log_path, result.stdout)
    append_to_log(log_path, result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")


def ensure_patched_ollama(logger: logging.Logger) -> Path:
    if BIN_PATH.exists():
        logger.info("Using cached patched Ollama binary at %s", BIN_PATH)
        return BIN_PATH

    logger.info("Building patched Ollama PR %s into %s", PATCH_REF, CACHE_ROOT)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    if SRC_DIR.exists():
        shutil.rmtree(SRC_DIR)

    run_and_log(
        ["git", "clone", "--depth", "1", "https://github.com/ollama/ollama.git", str(SRC_DIR)],
        cwd=REPO_ROOT,
        log_path=BUILD_LOG,
    )

    patch_resp = requests.get(PATCH_URL, timeout=60)
    patch_resp.raise_for_status()
    PATCH_FILE.write_text(patch_resp.text, encoding="utf-8")

    run_and_log(["git", "apply", str(PATCH_FILE)], cwd=SRC_DIR, log_path=BUILD_LOG)
    run_and_log(["go", "clean", "-cache"], cwd=SRC_DIR, log_path=BUILD_LOG)
    run_and_log(["go", "build", "-o", str(BIN_PATH), "."], cwd=SRC_DIR, log_path=BUILD_LOG)
    logger.info("Patched Ollama build completed")
    return BIN_PATH


def wait_for_http(url: str, timeout_seconds: int = 60) -> None:
    deadline = time.time() + timeout_seconds
    last_error = "unknown"
    while time.time() < deadline:
        try:
            resp = requests.get(url, timeout=3)
            resp.raise_for_status()
            return
        except Exception as exc:
            last_error = str(exc)
            time.sleep(1)
    raise RuntimeError(f"Timed out waiting for {url}: {last_error}")


def start_patched_server(binary_path: Path, port: int, logger: logging.Logger):
    serve_log_fh = SERVE_LOG.open("a", encoding="utf-8")
    env = os.environ.copy()
    env["OLLAMA_HOST"] = f"127.0.0.1:{port}"
    proc = subprocess.Popen(
        [str(binary_path), "serve"],
        cwd=str(SRC_DIR),
        env=env,
        stdout=serve_log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        wait_for_http(f"http://127.0.0.1:{port}/api/version", timeout_seconds=30)
        logger.info("Patched Ollama server ready on 127.0.0.1:%s", port)
        return proc, serve_log_fh
    except Exception:
        serve_log_fh.flush()
        serve_log_fh.close()
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        raise


def stop_process_group(proc: subprocess.Popen[bytes | str] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=10)
    except Exception:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run_stage2(env: dict[str, str]) -> int:
    cmd = [sys.executable, str(STAGE2_SCRIPT), "--config", str(CONFIG_PATH.relative_to(REPO_ROOT))]
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=False)
    return result.returncode


def main() -> int:
    logger = setup_logger()
    cfg = load_config(CONFIG_PATH)
    local_cfg = cfg.get("headline_filter", {}).get("local", {})
    provider = str(local_cfg.get("provider", "ollama")).strip().lower()
    base_url = normalized_base_url(local_cfg.get("base_url"))

    if provider != "ollama":
        logger.info("Local provider is %s; running Stage 2 directly", provider)
        return run_stage2(os.environ.copy())

    if not base_url or not base_url.endswith(":11434"):
        logger.info("Local Ollama base URL is %s; running Stage 2 directly", base_url)
        return run_stage2(os.environ.copy())

    binary_path = ensure_patched_ollama(logger)
    proc = None
    serve_log_fh = None
    try:
        proc, serve_log_fh = start_patched_server(binary_path, PATCH_PORT, logger)
        run_env = os.environ.copy()
        run_env["LOCAL_LLM_BASE_URL"] = f"http://127.0.0.1:{PATCH_PORT}"
        return run_stage2(run_env)
    finally:
        stop_process_group(proc)
        if serve_log_fh is not None:
            serve_log_fh.flush()
            serve_log_fh.close()


if __name__ == "__main__":
    raise SystemExit(main())
