"""Google Scholar profile scraping + caching for the TUI in tools/manage.py.

Profile stats (name, affiliation, citation counts, h-index, citations-per-year,
papers) come from Scholar's `/citations?user=` page. Recent citations come from
OpenAlex (see tools/openalex.py) — Scholar's `/scholar?cites=` endpoint CAPTCHAs
non-browser clients aggressively and isn't worth scraping.

Pure data layer — no Textual / UI dependencies. Uses `urllib` from the stdlib
plus `beautifulsoup4` for the profile-page HTML.

Public API:
    fetch_profile(user_id, ...) -> Profile
    load_cache() -> Profile | None
    save_cache(profile) -> None
    load_config() -> dict
    save_config(cfg) -> None
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
CACHE_PATH = TOOLS_DIR / "scholar_cache.json"
CONFIG_PATH = TOOLS_DIR / "scholar_config.json"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) "
    "Version/17.6 Safari/605.1.15"
)

SCHOLAR_BASE = "https://scholar.google.com"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Paper:
    title: str
    authors: str
    venue: str
    year: int | None
    cited_by: int
    cited_by_url: str | None
    paper_url: str | None
    # OpenAlex-derived non-self citation count. None means "not yet resolved"
    # (either OpenAlex doesn't know this paper, or the refresh hasn't run yet).
    cited_by_non_self: int | None = None


@dataclass
class Citation:
    title: str
    authors: str
    venue: str
    year: int | None
    citing_paper_url: str | None
    of_paper_title: str  # which of the user's papers this citation cites
    # ISO publication date ("YYYY-MM-DD") from OpenAlex when available.
    date_iso: str | None = None
    # True iff the citing work shares an author with the profile owner (per
    # OpenAlex author ids). Default False keeps old cached entries loadable.
    is_self_citation: bool = False


@dataclass
class Profile:
    user_id: str
    name: str
    affiliation: str
    interests: list[str]
    total_citations: int
    citations_5y: int
    h_index: int
    h_index_5y: int
    i10_index: int
    i10_index_5y: int
    citations_per_year: dict[int, int]
    papers: list[Paper] = field(default_factory=list)
    recent_citations: list[Citation] = field(default_factory=list)
    fetched_at: float = 0.0
    # OpenAlex-derived non-self stats. Defaults represent "unresolved".
    total_citations_non_self: int | None = None
    citations_5y_non_self: int | None = None
    h_index_non_self: int | None = None
    i10_index_non_self: int | None = None
    citations_per_year_non_self: dict[int, int] = field(default_factory=dict)


class ScholarError(RuntimeError):
    """Raised when scraping fails (block, parse error, network)."""


class CaptchaError(ScholarError):
    """Scholar served a CAPTCHA / 'unusual traffic' page. Wait and retry."""


# ---------------------------------------------------------------------------
# Cache + config
# ---------------------------------------------------------------------------


def load_cache() -> Profile | None:
    if not CACHE_PATH.exists():
        return None
    try:
        d = json.loads(CACHE_PATH.read_text())
        papers = [Paper(**p) for p in d.pop("papers", [])]
        cites = [Citation(**c) for c in d.pop("recent_citations", [])]
        d["citations_per_year"] = {
            int(k): int(v) for k, v in d.get("citations_per_year", {}).items()
        }
        d["citations_per_year_non_self"] = {
            int(k): int(v) for k, v in d.get("citations_per_year_non_self", {}).items()
        }
        return Profile(papers=papers, recent_citations=cites, **d)
    except Exception:
        return None


def save_cache(profile: Profile) -> None:
    CACHE_PATH.write_text(json.dumps(asdict(profile), indent=2, ensure_ascii=False))


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(CONFIG_PATH.read_text())
    except Exception:
        return {}


def save_config(cfg: dict) -> None:
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def _http_get(url: str, *, timeout: float = 25.0) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;q=0.9,"
                "image/avif,image/webp,*/*;q=0.8"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": SCHOLAR_BASE + "/",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
    except urllib.error.HTTPError as e:
        if e.code == 429:
            raise CaptchaError(
                "Google Scholar returned 429 (rate-limited). Wait ~1 h."
            ) from e
        raise ScholarError(f"HTTP {e.code} fetching {url}") from e
    except urllib.error.URLError as e:
        raise ScholarError(f"network error fetching {url}: {e.reason}") from e
    body = data.decode("utf-8", errors="replace")
    lower = body.lower()
    if (
        "our systems have detected unusual traffic" in lower
        or "id=\"gs_captcha_ccl\"" in lower
        or "please show you&#39;re not a robot" in lower
        or "id=\"recaptcha\"" in lower
    ):
        raise CaptchaError(
            "Google Scholar served a CAPTCHA on the profile page. "
            "Wait ~1 h for the IP block to expire."
        )
    return body


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _int(s: str) -> int:
    try:
        return int(re.sub(r"[^\d]", "", s) or "0")
    except ValueError:
        return 0


def _abs(url: str | None) -> str | None:
    if not url:
        return None
    if url.startswith("http"):
        return url
    if url.startswith("/"):
        return SCHOLAR_BASE + url
    return SCHOLAR_BASE + "/" + url


def _parse_profile_page(html: str, user_id: str) -> Profile:
    try:
        from bs4 import BeautifulSoup
    except ImportError as e:  # pragma: no cover - import-time error
        raise ScholarError(
            "beautifulsoup4 is required. Install with: "
            "pip install -r tools/requirements.txt"
        ) from e

    soup = BeautifulSoup(html, "html.parser")

    name_el = soup.find(id="gsc_prf_in")
    affil_el = soup.find("div", class_="gsc_prf_il")
    if name_el is None:
        raise ScholarError(
            "Could not find profile name — markup may have changed, or the user "
            "id is wrong."
        )
    name = name_el.get_text(strip=True)
    affiliation = affil_el.get_text(strip=True) if affil_el else ""
    interests = [a.get_text(strip=True) for a in soup.find_all("a", class_="gsc_prf_inta")]

    cited_total = cited_5y = h = h5 = i10 = i10_5 = 0
    stats_table = soup.find("table", id="gsc_rsb_st")
    if stats_table:
        rows = stats_table.find_all("tr")
        # Header row + 3 data rows: Citations / h-index / i10-index
        data_rows = [r for r in rows if r.find_all("td")]
        cells = [r.find_all("td") for r in data_rows]
        if len(cells) >= 3 and all(len(c) >= 3 for c in cells[:3]):
            cited_total = _int(cells[0][1].get_text())
            cited_5y = _int(cells[0][2].get_text())
            h = _int(cells[1][1].get_text())
            h5 = _int(cells[1][2].get_text())
            i10 = _int(cells[2][1].get_text())
            i10_5 = _int(cells[2][2].get_text())

    years = [_int(s.get_text()) for s in soup.find_all("span", class_="gsc_g_t")]
    counts = [_int(s.get_text()) for s in soup.find_all("span", class_="gsc_g_al")]
    cpy: dict[int, int] = {}
    for y, c in zip(years, counts):
        if y > 0:
            cpy[y] = c

    papers: list[Paper] = []
    for row in soup.find_all("tr", class_="gsc_a_tr"):
        a = row.find("a", class_="gsc_a_at")
        title = a.get_text(strip=True) if a else ""
        paper_url = _abs(a.get("href")) if a else None
        details = row.find_all("div", class_="gs_gray")
        authors = details[0].get_text(strip=True) if details else ""
        venue = details[1].get_text(strip=True) if len(details) > 1 else ""
        cb_a = row.find("a", class_="gsc_a_ac")
        cb_text = cb_a.get_text(strip=True) if cb_a else ""
        cb_count = int(cb_text) if cb_text.isdigit() else 0
        cb_url = _abs(cb_a.get("href")) if cb_a and cb_a.get("href") else None
        y_el = row.find("span", class_="gsc_a_h")
        y_text = y_el.get_text(strip=True) if y_el else ""
        year = int(y_text) if y_text.isdigit() else None
        papers.append(
            Paper(
                title=title,
                authors=authors,
                venue=venue,
                year=year,
                cited_by=cb_count,
                cited_by_url=cb_url,
                paper_url=paper_url,
            )
        )

    return Profile(
        user_id=user_id,
        name=name,
        affiliation=affiliation,
        interests=interests,
        total_citations=cited_total,
        citations_5y=cited_5y,
        h_index=h,
        h_index_5y=h5,
        i10_index=i10,
        i10_index_5y=i10_5,
        citations_per_year=cpy,
        papers=papers,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _h_index(counts: list[int]) -> int:
    h = 0
    for i, c in enumerate(sorted(counts, reverse=True), 1):
        if c >= i:
            h = i
        else:
            break
    return h


def _resolve_openalex_author_id(profile: Profile, *, polite_email: str | None) -> str | None:
    """Find the user's OpenAlex author id via one of their resolved papers.

    Cached in scholar_config.json under "openalex_author_id" so this only
    runs once.
    """
    try:
        import openalex as oa
    except ImportError:
        return None
    cfg = load_config()
    cached = cfg.get("openalex_author_id")
    if cached:
        return cached
    titles = [p.title for p in profile.papers if p.title]
    if not titles:
        return None
    title_to_id = oa.resolve_paper_titles(titles, polite_email=polite_email)
    if not title_to_id:
        return None
    work_id = next(iter(title_to_id.values()))
    author_id = oa.resolve_author_id(
        work_id, profile.name, polite_email=polite_email
    )
    if author_id:
        cfg["openalex_author_id"] = author_id
        save_config(cfg)
    return author_id


def _populate_non_self_stats(
    profile: Profile,
    *,
    polite_email: str | None,
    author_id: str | None = None,
) -> None:
    """Compute and attach OpenAlex-derived non-self statistics to `profile`.

    Best-effort: on any failure the non-self fields stay at their defaults
    so the rest of the refresh still succeeds.
    """
    try:
        import openalex as oa
    except ImportError:
        return
    if author_id is None:
        author_id = _resolve_openalex_author_id(profile, polite_email=polite_email)
    if not author_id:
        return
    titles = [p.title for p in profile.papers if p.title]
    title_to_id = oa.resolve_paper_titles(titles, polite_email=polite_email)
    if not title_to_id:
        return
    per_work = oa.fetch_non_self_per_work(
        list(title_to_id.values()), author_id, polite_email=polite_email
    )
    counts: list[int] = []
    per_year_total: dict[int, int] = {}
    for paper in profile.papers:
        wid = title_to_id.get(paper.title)
        if not wid or wid not in per_work:
            paper.cited_by_non_self = None
            continue
        nsw = per_work[wid]
        paper.cited_by_non_self = nsw.cited_by
        counts.append(nsw.cited_by)
        for y, c in nsw.per_year.items():
            per_year_total[y] = per_year_total.get(y, 0) + c
    if not counts:
        return
    profile.total_citations_non_self = sum(counts)
    profile.h_index_non_self = _h_index(counts)
    profile.i10_index_non_self = sum(1 for c in counts if c >= 10)
    profile.citations_per_year_non_self = per_year_total
    if per_year_total:
        last_year = max(per_year_total)
        profile.citations_5y_non_self = sum(
            v for y, v in per_year_total.items() if y > last_year - 5
        )


def _fetch_citations_openalex(
    profile: Profile,
    *,
    polite_email: str | None,
    limit: int,
    author_id: str | None = None,
) -> list[Citation]:
    """Source recent citations from OpenAlex instead of Scholar."""
    try:
        import openalex as oa  # sibling module
    except ImportError as e:
        raise ScholarError(
            "tools/openalex.py is missing — should be next to scholar.py"
        ) from e

    titles = [p.title for p in profile.papers if p.title]
    title_to_id = oa.resolve_paper_titles(titles, polite_email=polite_email)
    if not title_to_id:
        return []
    id_to_title = {oa_id: title for title, oa_id in title_to_id.items()}
    works = oa.fetch_recent_citations(
        list(id_to_title.keys()), limit=limit, polite_email=polite_email
    )
    out: list[Citation] = []
    for w in works:
        # Which of *our* papers does this work cite? Pick the first match.
        ours = next((wid for wid in w.referenced_work_ids if wid in id_to_title), None)
        if ours is None:
            continue
        out.append(
            Citation(
                title=w.title,
                authors=w.authors,
                venue=w.venue,
                year=w.year,
                citing_paper_url=(
                    w.doi if w.doi and w.doi.startswith("http")
                    else f"https://openalex.org/{w.id}"
                ),
                of_paper_title=id_to_title[ours],
                date_iso=w.publication_date,
                is_self_citation=bool(author_id and author_id in w.author_ids),
            )
        )
    return out


def fetch_profile(
    user_id: str,
    *,
    fetch_citations: bool = True,
    polite_email: str | None = None,
) -> Profile:
    """Scrape the Scholar profile page, then resolve recent citations via OpenAlex.

    `fetch_citations=False` skips the OpenAlex step (much faster — useful for
    a stats-only refresh).  `polite_email` is forwarded to OpenAlex's polite
    pool for friendlier rate limits.
    """
    if not user_id or not re.fullmatch(r"[A-Za-z0-9_-]{6,20}", user_id):
        raise ScholarError(f"invalid user id: {user_id!r}")

    profile_url = (
        f"{SCHOLAR_BASE}/citations?user={urllib.parse.quote(user_id)}"
        "&hl=en&pagesize=100"
    )
    profile = _parse_profile_page(_http_get(profile_url), user_id)

    if fetch_citations:
        # Resolve once and share between citations + non-self stats so we
        # only pay for the title→work + work→author lookups one time per run.
        author_id = _resolve_openalex_author_id(profile, polite_email=polite_email)
        profile.recent_citations = _fetch_citations_openalex(
            profile, polite_email=polite_email, limit=30, author_id=author_id
        )
        _populate_non_self_stats(
            profile, polite_email=polite_email, author_id=author_id
        )

    profile.fetched_at = time.time()
    return profile


# ---------------------------------------------------------------------------
# CLI helper (mostly for one-off debugging)
# ---------------------------------------------------------------------------


def _cli() -> int:  # pragma: no cover - manual debugging entry
    import argparse

    parser = argparse.ArgumentParser(description="Fetch Google Scholar profile.")
    parser.add_argument("--user", required=True, help="Scholar user id")
    parser.add_argument(
        "--no-citations", action="store_true", help="skip the OpenAlex citation lookup"
    )
    args = parser.parse_args()
    p = fetch_profile(args.user, fetch_citations=not args.no_citations)
    print(f"{p.name} — {p.affiliation}")
    print(
        f"citations={p.total_citations} (5y={p.citations_5y})  "
        f"h={p.h_index} (5y={p.h_index_5y})  "
        f"i10={p.i10_index} (5y={p.i10_index_5y})"
    )
    print(f"papers={len(p.papers)}, recent_citations={len(p.recent_citations)}")
    save_cache(p)
    print(f"wrote {CACHE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
