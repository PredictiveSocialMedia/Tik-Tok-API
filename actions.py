"""
TikTok Scraper Actions

Helper functions used by the main scraper, built against the actual
TikTok desktop DOM (2024-2026 layout).  Covers:

- Chrome driver setup
- Page loading and rehydration-JSON extraction
- Navigating TikTok's __DEFAULT_SCOPE__ JSON tree
- DOM interaction (overlays, comment sidebar, scrolling)
- Parsing video items, comments, and counts
"""

import json
import re
import time
from typing import Any, Optional

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def parse_count(text: str) -> int:
    """
    Turn human-readable counts like '28.2K', '1.9M', '9339' into ints.
    Also handles aria-label patterns like "276.8K likes".
    """
    if not text:
        return 0
    text = text.replace(",", "").strip()
    m = re.match(r"([\d.]+)\s*([KkMmBb])?", text)
    if not m:
        return 0
    try:
        value = float(m.group(1))
        suffix = (m.group(2) or "").upper()
        if suffix == "K":
            value *= 1_000
        elif suffix == "M":
            value *= 1_000_000
        elif suffix == "B":
            value *= 1_000_000_000
        return int(value)
    except (ValueError, TypeError):
        return 0


def safe_get(obj: Any, *keys: str, default: Any = None) -> Any:
    """Drill into nested dicts: ``safe_get(d, 'a', 'b', 'c')``."""
    for key in keys:
        if not isinstance(obj, dict) or key not in obj:
            return default
        obj = obj[key]
    return obj


def find_in_dict(obj: Any, key: str, results: Optional[list] = None) -> list:
    """Recursively collect every value for *key* in nested dicts / lists."""
    if results is None:
        results = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                results.append(v)
            find_in_dict(v, key, results)
    elif isinstance(obj, list):
        for item in obj:
            find_in_dict(item, key, results)
    return results


# ---------------------------------------------------------------------------
# Driver setup
# ---------------------------------------------------------------------------


def create_driver(headless: bool = True) -> webdriver.Chrome:
    """Return a Chrome WebDriver configured for TikTok scraping."""
    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument(
        "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    driver = webdriver.Chrome(options=opts)
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"},
    )
    return driver


# ---------------------------------------------------------------------------
# Page load & JSON extraction
# ---------------------------------------------------------------------------


def load_tiktok_video_page(
    driver: webdriver.Chrome,
    video_url: str,
    wait_seconds: int = 10,
    solve_captcha: bool = True,
) -> str:
    """
    Navigate to a TikTok video URL and return the rendered page source.

    If *solve_captcha* is ``True`` (default) and the ``object_selection_captcha`` package is
    installed, any blocking CAPTCHA will be detected and solved automatically
    before returning the page source.
    """
    driver.get(video_url)
    try:
        WebDriverWait(driver, wait_seconds).until(
            EC.presence_of_element_located(
                (By.ID, "__UNIVERSAL_DATA_FOR_REHYDRATION__")
            )
        )
    except Exception:
        # Fallback: wait for any script tag
        WebDriverWait(driver, wait_seconds).until(
            EC.presence_of_element_located((By.TAG_NAME, "script"))
        )
    time.sleep(2)

    # ── CAPTCHA handling ─────────────────────────────────────────────
    if solve_captcha:
        try:
            from object_selection_captcha import handle_captcha

            solved = handle_captcha(driver, max_attempts=3)
            if solved:
                # Re-wait for the page content after the CAPTCHA clears
                time.sleep(1)
        except ImportError:
            pass  # object_selection_captcha not available – skip silently

    return driver.page_source


def extract_rehydration_json(html: str) -> Optional[dict]:
    """
    Pull the ``__UNIVERSAL_DATA_FOR_REHYDRATION__`` JSON blob from the page.
    Returns the parsed dict or ``None``.
    """
    soup = BeautifulSoup(html, "html.parser")
    script = soup.find("script", id="__UNIVERSAL_DATA_FOR_REHYDRATION__")
    if not script or not script.string:
        return None
    try:
        return json.loads(script.string)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Navigate TikTok's rehydration JSON
# ---------------------------------------------------------------------------


def get_video_item_payload(data: dict) -> Optional[dict]:
    """
    Locate the video-detail payload inside the rehydration JSON.

    Known path (2024-2026 desktop):
        __DEFAULT_SCOPE__  →  webapp.video-detail  →  itemInfo  →  itemStruct

    Falls back to recursive search if the path changes.
    """
    # --- Primary path: __DEFAULT_SCOPE__ at top level ---
    default_scope = data.get("__DEFAULT_SCOPE__")
    if isinstance(default_scope, dict):
        video_detail = default_scope.get("webapp.video-detail")
        if isinstance(video_detail, dict):
            if "itemInfo" in video_detail:
                return video_detail

    # --- Legacy paths (older page versions) ---
    for key in ("defaultProps", "props", "pageProps"):
        payload = data.get(key)
        if not isinstance(payload, dict):
            continue
        for scope in ("__DEFAULT_SCOPE__", "webapp.video-detail", "videoDetail"):
            scope_data = payload.get(scope)
            if isinstance(scope_data, dict) and "itemInfo" in scope_data:
                return scope_data
        if "itemInfo" in payload:
            return payload

    # --- Fallback: brute-force recursive search ---
    for info in find_in_dict(data, "itemInfo"):
        if isinstance(info, dict) and "itemStruct" in info:
            return {"itemInfo": info}
    return None


def get_comments_from_data(data: dict) -> list[dict]:
    """Try to extract pre-loaded comments from the rehydration JSON."""
    # The desktop page rarely includes comments in JSON; they load async.
    for key in ("comments", "commentList"):
        for result in find_in_dict(data, key):
            if isinstance(result, list) and result:
                if isinstance(result[0], dict) and ("text" in result[0] or "cid" in result[0]):
                    return result
    return []


# ---------------------------------------------------------------------------
# DOM interaction – overlays, comment sidebar, scrolling
# ---------------------------------------------------------------------------


def dismiss_overlays(driver: webdriver.Chrome) -> None:
    """Attempt to close cookie banners or other overlays that block clicks."""
    cookie_labels = (
        "Allow all",
        "Allow All",
        "Accept all cookies",
        "Accept All",
        "Decline optional cookies",
        "Reject all",
    )
    for label in cookie_labels:
        try:
            btn = driver.find_element(
                By.XPATH, f"//button[normalize-space()='{label}']"
            )
            btn.click()
            time.sleep(0.5)
            return
        except Exception:
            continue


def click_comment_icon(driver: webdriver.Chrome, timeout: int = 6) -> None:
    """
    Click the comment action button to open the comment sidebar / tab.
    Matches the TikTok desktop DOM where the button has an aria-label
    starting with 'Read or add comments' or contains the comment-icon span.
    """
    try:
        btn = WebDriverWait(driver, timeout).until(
            EC.element_to_be_clickable(
                (
                    By.XPATH,
                    "//button[starts-with(@aria-label,'Read or add comments')]"
                    " | //button[.//span[@data-e2e='comment-icon']]",
                )
            )
        )
        btn.click()
        # Wait for the comment list container
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "div[class*='DivCommentListContainer']")
            )
        )
        time.sleep(0.5)
    except Exception:
        pass


def scroll_to_load_comments(driver: webdriver.Chrome, wait_seconds: int = 6) -> None:
    """Open comments and scroll to trigger lazy-loaded comment elements."""
    try:
        dismiss_overlays(driver)
        click_comment_icon(driver)
        for offset in (600, 800, 400):
            driver.execute_script(f"window.scrollBy(0, {offset});")
            time.sleep(1.2)
        try:
            WebDriverWait(driver, wait_seconds).until(
                EC.presence_of_element_located(
                    (
                        By.CSS_SELECTOR,
                        "div[class*='DivCommentListContainer'],"
                        "div[class*='DivCommentObjectWrapper']",
                    )
                )
            )
        except Exception:
            pass
        time.sleep(1)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# DOM comment scraping
# ---------------------------------------------------------------------------


def get_comments_from_dom(driver: webdriver.Chrome, max_comments: int) -> list[dict]:
    """
    Scrape visible comments from the rendered DOM.

    Each comment wrapper (``DivCommentObjectWrapper``) contains:
    - ``data-e2e="comment-username-1"`` → username link
    - ``data-e2e="comment-level-1"`` → comment text
    - ``DivLikeContainer`` with aria-label → like count
    - ``DivCommentSubContentWrapper`` → date
    - Sibling ``DivReplyContainer`` with ``TUXButton-label`` → reply count
    """
    scroll_to_load_comments(driver, wait_seconds=5)

    comments: list[dict] = []
    seen: set[str] = set()

    # ── Locate comment wrappers ──────────────────────────────────────
    wrappers = driver.find_elements(
        By.CSS_SELECTOR, "div[class*='DivCommentObjectWrapper']"
    )
    if not wrappers:
        wrappers = driver.find_elements(
            By.CSS_SELECTOR, "div[class*='DivCommentItemWrapper']"
        )

    for wrapper in wrappers:
        if len(comments) >= max_comments:
            break
        try:
            # ── Comment text ─────────────────────────────────────
            text_el = wrapper.find_elements(
                By.CSS_SELECTOR, "[data-e2e='comment-level-1']"
            )
            if not text_el:
                continue
            text = (text_el[0].text or "").strip()
            if len(text) < 2 or len(text) > 2000 or text in seen:
                continue
            seen.add(text)

            # ── Username ─────────────────────────────────────────
            username = ""
            try:
                user_el = wrapper.find_elements(
                    By.CSS_SELECTOR, "[data-e2e='comment-username-1'] a p"
                )
                if user_el:
                    username = (user_el[0].text or "").strip()
            except Exception:
                pass

            # ── Like count ───────────────────────────────────────
            like_count = 0
            try:
                like_container = wrapper.find_elements(
                    By.CSS_SELECTOR, "div[class*='DivLikeContainer']"
                )
                if like_container:
                    # Try aria-label first  ("Like video\n28.2K likes")
                    aria = like_container[0].get_attribute("aria-label") or ""
                    m = re.search(r"([\d.,]+[KkMmBb]?)\s*likes?", aria)
                    if m:
                        like_count = parse_count(m.group(1))
                    else:
                        # Fallback: span text inside container
                        spans = like_container[0].find_elements(By.CSS_SELECTOR, "span")
                        if spans:
                            like_count = parse_count(spans[-1].text)
            except Exception:
                pass

            # ── Reply count ──────────────────────────────────────
            reply_count = 0
            try:
                reply_els = wrapper.find_elements(
                    By.XPATH,
                    "following-sibling::div[contains(@class,'DivReplyContainer')][1]"
                    "//div[contains(@class,'TUXButton-label')]",
                )
                if reply_els:
                    label = (reply_els[0].text or "").strip()
                    m = re.search(r"View\s+([\d,]+)\s+repl", label)
                    if m:
                        reply_count = int(m.group(1).replace(",", ""))
            except Exception:
                pass

            # ── Date ─────────────────────────────────────────────
            date = ""
            try:
                date_els = wrapper.find_elements(
                    By.CSS_SELECTOR,
                    "div[class*='DivCommentSubContentWrapper'] span",
                )
                if date_els:
                    date = (date_els[0].text or "").strip()
            except Exception:
                pass

            comments.append(
                {
                    "text": text,
                    "username": username,
                    "likeCount": like_count,
                    "replyCount": reply_count,
                    "date": date,
                    "from_dom": True,
                }
            )
        except Exception:
            continue

    # ── Fallback: bare comment-level-1 elements ──────────────────────
    if not comments:
        try:
            els = driver.find_elements(
                By.CSS_SELECTOR, "[data-e2e='comment-level-1']"
            )
            for el in els[: max_comments * 2]:
                if len(comments) >= max_comments:
                    break
                t = (el.text or "").strip()
                if 2 < len(t) < 2000 and t not in seen:
                    seen.add(t)
                    comments.append({"text": t, "from_dom": True})
        except Exception:
            pass

    return comments[:max_comments]


# ---------------------------------------------------------------------------
# Structured data parsing
# ---------------------------------------------------------------------------


def parse_video_item(item: dict) -> dict:
    """
    Convert a TikTok ``itemStruct`` dict into a clean, flat metadata dict.

    Handles both ``stats`` (int values) and ``statsV2`` (string values)
    that TikTok alternates between.
    """
    author = item.get("author") or {}
    music = item.get("music") or {}
    video = item.get("video") or {}

    # ── Stats (prefer statsV2 strings, fall back to stats ints) ──────
    stats = item.get("statsV2") or item.get("stats") or {}
    play_count = parse_count(str(stats.get("playCount", 0)))
    like_count = parse_count(str(stats.get("diggCount") or stats.get("likeCount", 0)))
    comment_count = parse_count(str(stats.get("commentCount", 0)))
    share_count = parse_count(str(stats.get("shareCount", 0)))
    collect_count = parse_count(str(stats.get("collectCount", 0)))
    repost_count = parse_count(str(stats.get("repostCount", 0)))

    # ── Video URLs ───────────────────────────────────────────────────
    play_addr = video.get("playAddr") or ""
    if not play_addr:
        # bitrateInfo[0] often has the best URL list
        bitrate_info = video.get("bitrateInfo") or []
        if bitrate_info:
            url_list = safe_get(bitrate_info[0], "PlayAddr", "UrlList", default=[])
            if url_list:
                play_addr = url_list[0]
    if not play_addr:
        play_addr = video.get("downloadAddr") or ""

    download_addr = video.get("downloadAddr") or play_addr

    # ── Music / audio ────────────────────────────────────────────────
    music_url = music.get("playUrl") or ""

    # ── Hashtags / challenges ────────────────────────────────────────
    challenges = item.get("challenges") or []
    hashtags = [
        {"id": c.get("id"), "title": c.get("title")}
        for c in challenges
        if isinstance(c, dict) and c.get("title")
    ]

    # ── Author stats ─────────────────────────────────────────────────
    author_stats_raw = item.get("authorStatsV2") or item.get("authorStats") or {}

    return {
        "id": item.get("id"),
        "caption": (item.get("desc") or "").strip(),
        "createTime": item.get("createTime"),
        "author": {
            "id": author.get("id"),
            "uniqueId": author.get("uniqueId"),
            "nickname": author.get("nickname"),
            "verified": author.get("verified", False),
            "signature": (author.get("signature") or "").strip(),
            "avatarThumb": author.get("avatarThumb"),
        },
        "stats": {
            "playCount": play_count,
            "likeCount": like_count,
            "commentCount": comment_count,
            "shareCount": share_count,
            "collectCount": collect_count,
            "repostCount": repost_count,
        },
        "authorStats": {
            "followerCount": parse_count(str(author_stats_raw.get("followerCount", 0))),
            "followingCount": parse_count(str(author_stats_raw.get("followingCount", 0))),
            "heartCount": parse_count(str(author_stats_raw.get("heartCount", 0))),
            "videoCount": parse_count(str(author_stats_raw.get("videoCount", 0))),
        },
        "video": {
            "playUrl": play_addr,
            "downloadUrl": download_addr,
            "cover": video.get("cover") or video.get("originCover") or "",
            "duration": video.get("duration"),
            "width": video.get("width"),
            "height": video.get("height"),
            "ratio": video.get("ratio"),
            "format": video.get("format"),
        },
        "music": {
            "id": music.get("id"),
            "title": music.get("title") or "",
            "authorName": music.get("authorName") or "",
            "playUrl": music_url,
            "original": music.get("original", False),
            "duration": music.get("duration"),
        },
        "hashtags": hashtags,
        "diversificationLabels": item.get("diversificationLabels") or [],
        "locationCreated": item.get("locationCreated") or "",
    }


def parse_comment(raw: dict) -> dict:
    """Normalise a comment dict from the rehydration JSON."""
    text = raw.get("text") or safe_get(raw, "comment", "text") or ""
    if isinstance(text, dict):
        text = text.get("text", "")
    user = raw.get("user") or {}
    return {
        "id": raw.get("cid") or raw.get("id"),
        "text": text,
        "username": user.get("uniqueId") or user.get("unique_id") or "",
        "nickname": user.get("nickname") or "",
        "likeCount": raw.get("digg_count") or raw.get("diggCount") or raw.get("likeCount", 0),
        "replyCount": raw.get("reply_comment_total") or raw.get("replyCommentTotal", 0),
        "createTime": raw.get("create_time") or raw.get("createTime"),
    }
