"""OpenAlex API client — used by tools/manage.py to source recent citations
when Google Scholar CAPTCHAs the /scholar?cites= endpoint.

Public API:
    resolve_paper_titles(titles, polite_email=None) -> dict[str, str]
    fetch_recent_citations(work_ids, *, limit=30) -> list[OAWork]

OpenAlex is a free, no-auth scholarly database with comprehensive citation
data and real publication dates. Coverage of brand-new (last few weeks)
citations can lag Scholar slightly but is otherwise comparable.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

API_BASE = "https://api.openalex.org"
TOOLS_DIR = Path(__file__).resolve().parent
ID_CACHE_PATH = TOOLS_DIR / "openalex_id_cache.json"


class OpenAlexError(RuntimeError):
    """Raised on OpenAlex network / parse errors."""


@dataclass
class OAWork:
    """A citing work returned by fetch_recent_citations."""

    id: str  # bare W-id, e.g. "W2741809807"
    title: str
    authors: str  # joined display names, "A, B, C"
    venue: str
    year: int | None
    publication_date: str | None  # "YYYY-MM-DD"
    doi: str | None
    referenced_work_ids: list[str]  # bare W-ids this work cites


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def _http_json(url: str, *, polite_email: str | None = None, timeout: float = 20.0) -> dict:
    if polite_email:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}mailto={urllib.parse.quote(polite_email)}"
    headers = {
        "User-Agent": (
            f"uzerbinati.eu-scholar-watcher (mailto:{polite_email})"
            if polite_email
            else "uzerbinati.eu-scholar-watcher"
        ),
        "Accept": "application/json",
    }
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise OpenAlexError(f"HTTP {e.code} from OpenAlex: {url}") from e
    except urllib.error.URLError as e:
        raise OpenAlexError(f"network error: {e.reason}") from e
    except json.JSONDecodeError as e:
        raise OpenAlexError(f"OpenAlex returned non-JSON for {url}") from e


# ---------------------------------------------------------------------------
# Title → ID resolution (cached)
# ---------------------------------------------------------------------------


def _bare_id(id_url: str) -> str:
    """OpenAlex IDs come back as full URLs; we store the W-only form."""
    return id_url.rsplit("/", 1)[-1] if id_url else id_url


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w]", " ", s.lower())).strip()


def _word_overlap(a: str, b: str) -> float:
    aw, bw = set(a.split()), set(b.split())
    if not aw or not bw:
        return 0.0
    return len(aw & bw) / max(len(aw), len(bw))


def _search_one(title: str, *, polite_email: str | None) -> str | None:
    """Return the best-matching OpenAlex W-id for `title`, or None."""
    qs = urllib.parse.quote(title)
    data = _http_json(
        f"{API_BASE}/works?search={qs}&per-page=5", polite_email=polite_email
    )
    results = data.get("results") or []
    if not results:
        return None
    norm_target = _normalize(title)
    # 1. exact normalized-title match
    for r in results:
        if _normalize(r.get("title") or "") == norm_target:
            return _bare_id(r.get("id") or "")
    # 2. ≥75% word overlap on the top hit
    top = results[0]
    if _word_overlap(_normalize(top.get("title") or ""), norm_target) >= 0.75:
        return _bare_id(top.get("id") or "")
    return None


def _load_id_cache() -> dict[str, str]:
    if not ID_CACHE_PATH.exists():
        return {}
    try:
        return json.loads(ID_CACHE_PATH.read_text())
    except Exception:
        return {}


def _save_id_cache(d: dict[str, str]) -> None:
    ID_CACHE_PATH.write_text(json.dumps(d, indent=2, ensure_ascii=False))


def resolve_paper_titles(
    titles: list[str], *, polite_email: str | None = None
) -> dict[str, str]:
    """Map paper titles → OpenAlex W-ids, persisting results so we only hit
    the API once per never-seen title.

    Titles that resolve to nothing are cached as empty string so we don't
    retry them every refresh.
    """
    cache = _load_id_cache()
    out: dict[str, str] = {}
    dirty = False
    for title in titles:
        key = _normalize(title)
        if key in cache:
            if cache[key]:
                out[title] = cache[key]
            continue
        oa_id = _search_one(title, polite_email=polite_email)
        cache[key] = oa_id or ""
        dirty = True
        if oa_id:
            out[title] = oa_id
        time.sleep(0.1)  # OpenAlex allows 10/s anonymous; 0.1s is safe
    if dirty:
        _save_id_cache(cache)
    return out


# ---------------------------------------------------------------------------
# Citing-works query
# ---------------------------------------------------------------------------


def fetch_recent_citations(
    work_ids: list[str],
    *,
    limit: int = 30,
    polite_email: str | None = None,
) -> list[OAWork]:
    """Return up to `limit` works that cite any of `work_ids`, newest first."""
    if not work_ids:
        return []
    # OpenAlex supports OR within a single filter via the pipe syntax.
    # 50 W-ids comfortably fits in a URL.
    filter_value = "cites:" + "|".join(work_ids)
    url = (
        f"{API_BASE}/works"
        f"?filter={urllib.parse.quote(filter_value)}"
        f"&sort=publication_date:desc"
        f"&per-page={min(limit, 200)}"
        f"&select=id,title,authorships,primary_location,publication_date,"
        f"publication_year,doi,referenced_works"
    )
    data = _http_json(url, polite_email=polite_email)
    out: list[OAWork] = []
    for w in data.get("results") or []:
        authors = ", ".join(
            (a.get("author") or {}).get("display_name", "")
            for a in (w.get("authorships") or [])[:5]
            if (a.get("author") or {}).get("display_name")
        )
        primary = w.get("primary_location") or {}
        source = primary.get("source") or {}
        venue = source.get("display_name") or ""
        doi = w.get("doi")
        out.append(
            OAWork(
                id=_bare_id(w.get("id") or ""),
                title=w.get("title") or "",
                authors=authors,
                venue=venue,
                year=w.get("publication_year"),
                publication_date=w.get("publication_date"),
                doi=doi,
                referenced_work_ids=[_bare_id(r) for r in (w.get("referenced_works") or [])],
            )
        )
    return out
