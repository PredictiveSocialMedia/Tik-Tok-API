"""
Parser utilities: count parsing, safe dict access, JSON extraction.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from bs4 import BeautifulSoup


def parse_count(text: str) -> int:
    """Parse '28.2K', '1.9M', '9339' into int."""
    if not text:
        return 0
    text = str(text).replace(",", "").strip()
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
    """Drill into nested dicts."""
    for key in keys:
        if not isinstance(obj, dict) or key not in obj:
            return default
        obj = obj[key]
    return obj


def find_in_dict(obj: Any, key: str, results: Optional[list] = None) -> list:
    """Recursively collect every value for *key* in nested dicts/lists."""
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


def extract_rehydration_json(html: str) -> Optional[dict]:
    """Extract __UNIVERSAL_DATA_FOR_REHYDRATION__ from page HTML."""
    soup = BeautifulSoup(html, "html.parser")
    script = soup.find("script", id="__UNIVERSAL_DATA_FOR_REHYDRATION__")
    if not script or not script.string:
        return None
    try:
        return json.loads(script.string)
    except json.JSONDecodeError:
        return None


def get_video_item_payload(data: dict) -> Optional[dict]:
    """Locate itemInfo/itemStruct in rehydration JSON."""
    default_scope = data.get("__DEFAULT_SCOPE__")
    if isinstance(default_scope, dict):
        video_detail = default_scope.get("webapp.video-detail")
        if isinstance(video_detail, dict) and "itemInfo" in video_detail:
            return video_detail
    for key in ("defaultProps", "props", "pageProps"):
        payload = data.get(key)
        if not isinstance(payload, dict):
            continue
        for scope in ("__DEFAULT_SCOPE__", "webapp.video-detail", "videoDetail"):
            scope_data = payload.get(scope) if isinstance(payload, dict) else {}
            if isinstance(scope_data, dict) and "itemInfo" in scope_data:
                return scope_data
        if "itemInfo" in payload:
            return payload
    for info in find_in_dict(data, "itemInfo"):
        if isinstance(info, dict) and "itemStruct" in info:
            return {"itemInfo": info}
    return None
