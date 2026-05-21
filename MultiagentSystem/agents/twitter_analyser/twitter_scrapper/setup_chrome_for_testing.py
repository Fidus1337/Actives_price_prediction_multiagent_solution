"""
One-time bootstrap for Chrome for Testing (CfT).

Downloads a pinned, never-auto-updating Chrome binary + the exactly matching
chromedriver and unpacks them next to the scraper. This decouples scraping from
the system Chrome, which auto-updates in the background and otherwise drifts out
of sync with undetected_chromedriver's cached driver
(SessionNotCreatedException: "ChromeDriver only supports Chrome version N").

Setup:
    Run this once (downloads ~200 MB for your OS):
        python -m MultiagentSystem.agents.twitter_analyser.twitter_scrapper.setup_chrome_for_testing
    By default it installs into <this dir>/chrome-for-testing, which _init_driver()
    auto-detects with no extra config. (Setting CHROME_FOR_TESTING_DIR is optional —
    the Docker image sets it to /opt/chrome-for-testing and downloads CfT at build.)

Platform is auto-detected (win64 / linux64); override with --platform to prefetch
the other OS's binaries (e.g. linux64 from a Windows box).

Re-running is idempotent (skips download if binaries exist; pass --force to redo).
To bump the pinned version: change PINNED_VERSION (and the Docker CFT_VERSION arg),
re-run with --force.

Only stdlib is used so this works before any project deps are installed.
"""

import argparse
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

# Latest Chrome-for-Testing build on the 148 milestone (matches the local Chrome
# major so the existing chrome_profile / cookies stay compatible). chrome and
# chromedriver are published together for this exact version, so browser and
# driver are guaranteed to match. Keep in sync with the Dockerfile CFT_VERSION arg.
PINNED_VERSION = "148.0.7778.178"

_CFT_BASE = "https://storage.googleapis.com/chrome-for-testing-public"
_DEFAULT_DIR = Path(__file__).parent / "chrome-for-testing"

# platform slug -> (chrome binary name, chromedriver binary name)
_BINARY_NAMES = {
    "win64": ("chrome.exe", "chromedriver.exe"),
    "linux64": ("chrome", "chromedriver"),
}

LOG_TAG = "[cft-setup]"


def _detect_platform() -> str:
    if sys.platform.startswith("win"):
        return "win64"
    if sys.platform.startswith("linux"):
        return "linux64"
    raise RuntimeError(
        f"{LOG_TAG} unsupported platform: {sys.platform}. Pass --platform win64|linux64"
    )


def _download(url: str, dest: Path) -> None:
    print(f"{LOG_TAG} downloading {url}")
    with urllib.request.urlopen(url) as resp:  # noqa: S310 (trusted Google host)
        if resp.status != 200:
            raise RuntimeError(f"{url} -> HTTP {resp.status}")
        with open(dest, "wb") as f:
            shutil.copyfileobj(resp, f)
    print(f"{LOG_TAG}   saved -> {dest} ({dest.stat().st_size // (1024 * 1024)} MB)")


def _extract(zip_path: Path, target_dir: Path) -> None:
    """Extract preserving the zip's top folder (chrome-<plat> / chromedriver-<plat>)."""
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target_dir)


def setup(version: str, target_dir: Path, platform: str, force: bool) -> tuple[Path, Path]:
    if platform not in _BINARY_NAMES:
        raise RuntimeError(f"{LOG_TAG} unsupported --platform {platform}; use win64 or linux64")
    chrome_name, driver_name = _BINARY_NAMES[platform]
    chrome_dir = f"chrome-{platform}"
    driver_dir = f"chromedriver-{platform}"
    chrome_bin = target_dir / chrome_dir / chrome_name
    driver_bin = target_dir / driver_dir / driver_name

    if chrome_bin.exists() and driver_bin.exists() and not force:
        print(f"{LOG_TAG} already installed (use --force to reinstall):")
        print(f"{LOG_TAG}   {chrome_bin}")
        print(f"{LOG_TAG}   {driver_bin}")
        _write_version(target_dir, version)
        return chrome_bin, driver_bin

    # Clean stale top folders on a forced reinstall.
    if force:
        for sub in (chrome_dir, driver_dir):
            stale = target_dir / sub
            if stale.exists():
                shutil.rmtree(stale, ignore_errors=True)

    target_dir.mkdir(parents=True, exist_ok=True)

    chrome_url = f"{_CFT_BASE}/{version}/{platform}/chrome-{platform}.zip"
    driver_url = f"{_CFT_BASE}/{version}/{platform}/chromedriver-{platform}.zip"

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        chrome_zip = tmp_path / "chrome.zip"
        driver_zip = tmp_path / "chromedriver.zip"
        _download(chrome_url, chrome_zip)
        _download(driver_url, driver_zip)
        print(f"{LOG_TAG} extracting...")
        _extract(chrome_zip, target_dir)
        _extract(driver_zip, target_dir)

    if not chrome_bin.exists() or not driver_bin.exists():
        raise RuntimeError(
            f"{LOG_TAG} extraction finished but expected binaries are missing:\n"
            f"  {chrome_bin} (exists={chrome_bin.exists()})\n"
            f"  {driver_bin} (exists={driver_bin.exists()})"
        )

    # zip archives don't always carry the exec bit on POSIX.
    if platform == "linux64":
        for b in (chrome_bin, driver_bin):
            b.chmod(0o755)

    _write_version(target_dir, version)
    return chrome_bin, driver_bin


def _write_version(target_dir: Path, version: str) -> None:
    (target_dir / "version.txt").write_text(version, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Install Chrome for Testing.")
    parser.add_argument("--version", default=PINNED_VERSION, help=f"CfT version (default {PINNED_VERSION})")
    parser.add_argument("--dir", default=str(_DEFAULT_DIR), help="install directory")
    parser.add_argument(
        "--platform",
        default=_detect_platform(),
        choices=sorted(_BINARY_NAMES),
        help="target platform (default: auto-detected from this OS)",
    )
    parser.add_argument("--force", action="store_true", help="re-download even if installed")
    args = parser.parse_args()

    target_dir = Path(args.dir).resolve()
    chrome_bin, driver_bin = setup(args.version, target_dir, args.platform, args.force)

    print()
    print(f"{LOG_TAG} DONE. Chrome for Testing {args.version} ({args.platform}) ready:")
    print(f"{LOG_TAG}   chrome      : {chrome_bin}")
    print(f"{LOG_TAG}   chromedriver: {driver_bin}")
    print()
    print(f"{LOG_TAG} _init_driver() auto-detects this default location — no dev.env edit needed.")
    print(f"{LOG_TAG} To use a custom location, set: CHROME_FOR_TESTING_DIR={target_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
