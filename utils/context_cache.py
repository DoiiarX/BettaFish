"""
Persistent cache for structured context payloads.

The Streamlit facades can store the timeline/label graph that each agent
produces, so subsequent sessions (or other agents) can reuse the same context
without running the full pipeline again.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional

DEFAULT_CACHE_DIR = Path("logs") / "context_cache"


class ContextCache:
    """File-based cache keyed by the normalized query string."""

    def __init__(self, cache_dir: Optional[os.PathLike] = None):
        self.cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path_for_key(self, key: str) -> Path:
        normalized = (key or "").strip().lower()
        digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def load(self, key: str) -> Optional[Dict[str, Any]]:
        path = self._path_for_key(key)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except (json.JSONDecodeError, OSError):
            return None

    def save(self, key: str, payload: Dict[str, Any]) -> None:
        if not key:
            return
        path = self._path_for_key(key)
        data = payload or {}
        metadata = data.setdefault("metadata", {})
        metadata["cached_at"] = datetime.utcnow().isoformat()
        with path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
