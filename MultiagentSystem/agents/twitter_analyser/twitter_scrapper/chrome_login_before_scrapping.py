"""
Chrome driver initialization and Twitter authentication.

Creates an undetected Chrome browser, manages login via credentials from dev.env,
and persists session cookies for reuse between runs.

First run requires login (credentials from dev.env).
Subsequent runs reuse saved cookies from twitter_cookies.json.

Usage:
    python -m MultiagentSystem.agents.twitter_analyser.twitter_scrapper.chrome_login_before_scrapping --login
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
import undetected_chromedriver as uc
from selenium.common.exceptions import SessionNotCreatedException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
load_dotenv(_PROJECT_ROOT / "dev.env")

COOKIES_PATH = Path(__file__).parent / "twitter_cookies.json"
LOG_TAG = "[twitter_selenium]"


def _build_options(headless: bool) -> uc.ChromeOptions:
    """Build fresh ChromeOptions instance for every driver attempt."""
    options = uc.ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--lang=en-US")
    return options


def _cleanup_profile_locks(scraper_profile: Path) -> None:
    """Remove stale Chrome singleton lock files from scraper profile."""
    lock_candidates = list(scraper_profile.glob("Singleton*"))
    # Some Chrome builds may place singleton locks one level deeper.
    lock_candidates.extend(scraper_profile.glob("**/Singleton*"))
    removed = 0
    for lock_path in lock_candidates:
        try:
            if lock_path.is_file() or lock_path.is_symlink():
                lock_path.unlink()
                removed += 1
        except Exception as e:
            print(f"{LOG_TAG} WARNING: could not remove lock file {lock_path}: {e}")
    if removed:
        print(f"{LOG_TAG} Removed {removed} stale profile lock file(s)")


def _detect_chrome_major() -> Optional[int]:
    """Try to detect local Chrome major version on Windows."""
    if os.name != "nt":
        return None
    try:
        import winreg  # type: ignore

        reg_paths = [
            (winreg.HKEY_CURRENT_USER, r"Software\Google\Chrome\BLBeacon"),
            (winreg.HKEY_LOCAL_MACHINE, r"Software\Google\Chrome\BLBeacon"),
            (winreg.HKEY_LOCAL_MACHINE, r"Software\WOW6432Node\Google\Chrome\BLBeacon"),
        ]
        for hive, subkey in reg_paths:
            try:
                with winreg.OpenKey(hive, subkey) as key:
                    version, _ = winreg.QueryValueEx(key, "version")
                    if isinstance(version, str) and version.strip():
                        major_part = version.split(".", 1)[0]
                        return int(major_part)
            except Exception:
                continue
    except Exception:
        return None
    return None


def _init_driver(headless: bool = True) -> uc.Chrome:
    """Start undetected Chrome with a dedicated profile for scraping."""
    scraper_profile = Path(__file__).parent / "chrome_profile"
    scraper_profile.mkdir(exist_ok=True)
    _cleanup_profile_locks(scraper_profile)

    detected_major = _detect_chrome_major()
    if detected_major:
        print(f"{LOG_TAG} Detected local Chrome major={detected_major}")

    # Prefer detected local major to avoid hanging in uc auto-detection.
    try:
        if detected_major:
            return uc.Chrome(
                options=_build_options(headless),
                user_data_dir=str(scraper_profile),
                version_main=detected_major,
            )
        return uc.Chrome(
            options=_build_options(headless),
            user_data_dir=str(scraper_profile),
            version_main=146,
        )
    except SessionNotCreatedException as e:
        print(f"{LOG_TAG} WARNING: profile-based launch failed: {e}")
        if detected_major:
            print(f"{LOG_TAG} Retrying without custom user_data_dir (major={detected_major})...")
        else:
            print(f"{LOG_TAG} Retrying with pinned major=146 without custom user_data_dir...")
        try:
            if detected_major:
                return uc.Chrome(
                    options=_build_options(headless),
                    version_main=detected_major,
                )
            return uc.Chrome(
                options=_build_options(headless),
                version_main=146,
            )
        except SessionNotCreatedException as e2:
            print(f"{LOG_TAG} WARNING: version-pinned fallback failed: {e2}")
            print(f"{LOG_TAG} Retrying final fallback without version pin...")
            return uc.Chrome(options=_build_options(headless))


def _save_cookies(driver: uc.Chrome) -> None:
    """Save browser cookies to JSON file."""
    cookies = driver.get_cookies()
    COOKIES_PATH.write_text(json.dumps(cookies, indent=2), encoding="utf-8")
    print(f"{LOG_TAG} Cookies saved ({len(cookies)} items)")


def _load_cookies(driver: uc.Chrome) -> bool:
    """Load cookies from file into browser. Returns True if loaded."""
    if not COOKIES_PATH.exists():
        return False

    cookies = json.loads(COOKIES_PATH.read_text(encoding="utf-8"))
    if not cookies:
        return False

    driver.get("https://x.com")
    time.sleep(5)

    added = 0
    for cookie in cookies:
        # Normalize to Selenium-compatible fields only.
        # Cookie-Editor exports expirationDate/hostOnly/session/storeId
        # which driver.add_cookie() rejects silently.
        normalized: dict = {
            "name": cookie["name"],
            "value": cookie["value"],
            "domain": cookie.get("domain", ".x.com"),
            "path": cookie.get("path", "/"),
            "secure": bool(cookie.get("secure", False)),
            "httpOnly": bool(cookie.get("httpOnly", False)),
        }
        expiry = cookie.get("expirationDate") or cookie.get("expiry")
        if expiry:
            normalized["expiry"] = int(expiry)
        try:
            driver.add_cookie(normalized)
            added += 1
        except Exception as e:
            print(f"{LOG_TAG} WARNING: could not add cookie '{cookie.get('name')}': {e}")

    print(f"{LOG_TAG} Cookies loaded ({added}/{len(cookies)} applied)")
    return added > 0


def _is_logged_in(driver: uc.Chrome) -> bool:
    """Check if we're logged into Twitter."""
    driver.get("https://x.com/home")
    time.sleep(6)
    url = driver.current_url
    logged_in = "login" not in url and "i/flow" not in url
    print(f"{LOG_TAG} Login check: url={url}, logged_in={logged_in}")
    return logged_in


def _login(driver: uc.Chrome) -> bool:
    """Login to Twitter via UI. Returns True on success."""
    email = os.getenv("TWITTER_EMAIL", "")
    username = os.getenv("TWITTER_USERNAME", "")
    password = os.getenv("TWITTER_PASSWORD", "")

    if not email or not password:
        print(f"{LOG_TAG} ERROR: Set TWITTER_EMAIL and TWITTER_PASSWORD in dev.env")
        return False

    print(f"{LOG_TAG} Logging in with {email}...")
    driver.get("https://x.com/i/flow/login")
    time.sleep(5)

    try:
        wait = WebDriverWait(driver, 20)

        username_input = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'input[autocomplete="username"]'))
        )
        username_input.clear()
        username_input.send_keys(email)
        username_input.send_keys(Keys.RETURN)
        time.sleep(3)

        unusual_activity = driver.find_elements(By.CSS_SELECTOR, 'input[data-testid="ocfEnterTextTextInput"]')
        if unusual_activity:
            print(f"{LOG_TAG} Verification required, trying @{username}...")
            unusual_activity[0].send_keys(username)
            unusual_activity[0].send_keys(Keys.RETURN)
            time.sleep(3)

            unusual_activity_retry = driver.find_elements(By.CSS_SELECTOR, 'input[data-testid="ocfEnterTextTextInput"]')
            if unusual_activity_retry:
                print(f"{LOG_TAG} Username rejected, trying email...")
                unusual_activity_retry[0].clear()
                unusual_activity_retry[0].send_keys(email)
                unusual_activity_retry[0].send_keys(Keys.RETURN)
                time.sleep(3)

        password_input = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'input[name="password"]'))
        )
        password_input.clear()
        password_input.send_keys(password)
        password_input.send_keys(Keys.RETURN)
        time.sleep(5)

        url = driver.current_url
        print(f"{LOG_TAG} Post-login URL: {url}")
        if "home" in url:
            print(f"{LOG_TAG} Login successful")
            _save_cookies(driver)
            return True
        else:
            print(f"{LOG_TAG} Login may have failed. URL: {url}")
            _save_cookies(driver)
            return "login" not in url and "i/flow" not in url

    except Exception as e:
        print(f"{LOG_TAG} Login error: {e}")
        return False


def _ensure_logged_in(driver: uc.Chrome) -> bool:
    """Load cookies or login. Returns True if logged in."""
    if _load_cookies(driver) and _is_logged_in(driver):
        print(f"{LOG_TAG} Session restored from cookies")
        return True

    print(f"{LOG_TAG} Cookies expired or missing, logging in...")
    return _login(driver)


def create_driver(headless: bool = True) -> uc.Chrome:
    """Create and authenticate a Chrome driver for reuse across multiple accounts.

    Returns an authenticated driver. Caller is responsible for driver.quit().
    """
    driver = _init_driver(headless=headless)
    if not _ensure_logged_in(driver):
        driver.quit()
        raise RuntimeError("Could not log in to Twitter")
    return driver


def save_cookies_from_upload(cookies: list[dict]) -> dict:
    """Write externally-provided cookies to twitter_cookies.json.

    Intended for re-login without stopping the API: user exports cookies from
    their browser (DevTools / EditThisCookie) and uploads them via API endpoint.

    Returns same shape as check_twitter_auth().
    """
    COOKIES_PATH.write_text(json.dumps(cookies, indent=2), encoding="utf-8")
    print(f"{LOG_TAG} Cookies written from upload ({len(cookies)} items) → {COOKIES_PATH}")
    return check_twitter_auth()


def check_twitter_auth() -> dict:
    """Lightweight auth status check — no Chrome required.

    Inspects the cookies file and env credentials without launching a browser.

    Returns:
        cookies_exist       — twitter_cookies.json exists and is non-empty
        session_cookies_ok  — auth_token and ct0 (session cookies) are present
        credentials_configured — TWITTER_EMAIL and TWITTER_PASSWORD are set in env
        cookies_path        — absolute path to the cookies file
        cookies_count       — number of cookies in the file
    """
    cookies_exist = False
    cookies_count = 0
    session_cookies_ok = False

    if COOKIES_PATH.exists():
        try:
            cookies = json.loads(COOKIES_PATH.read_text(encoding="utf-8"))
            cookies_count = len(cookies)
            cookies_exist = cookies_count > 0
            names = {c.get("name", "") for c in cookies}
            session_cookies_ok = "auth_token" in names and "ct0" in names
        except Exception:
            pass

    credentials_configured = bool(
        os.getenv("TWITTER_EMAIL", "").strip()
        and os.getenv("TWITTER_PASSWORD", "").strip()
    )

    return {
        "cookies_exist": cookies_exist,
        "session_cookies_ok": session_cookies_ok,
        "credentials_configured": credentials_configured,
        "cookies_path": str(COOKIES_PATH),
        "cookies_count": cookies_count,
    }


def manual_login() -> None:
    """Open Chrome for manual login. User logs in, script saves cookies."""
    driver = _init_driver(headless=False)
    try:
        driver.get("https://x.com/i/flow/login")
        print(f"\n{LOG_TAG} Chrome opened. Log in to Twitter manually.")
        print(f"{LOG_TAG} After login (when you see the feed) press Enter here...")
        input()

        if "home" in driver.current_url or "login" not in driver.current_url:
            _save_cookies(driver)
            print(f"{LOG_TAG} Cookies saved! Scraper will now work without login.")
        else:
            print(f"{LOG_TAG} Login seems to have failed. URL: {driver.current_url}")
    finally:
        driver.quit()


if __name__ == "__main__":
    if "--login" in sys.argv:
        manual_login()
