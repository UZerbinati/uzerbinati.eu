"""TUI for managing uzerbinati.eu papers, proceedings, preprints, and talks.

Run from the repo root:
    python tools/manage.py             # launch TUI
    python tools/manage.py --check     # parse + re-emit, fail on non-empty diff
    python tools/manage.py --rebuild   # rebuild category.md and exit
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


REPO = Path(__file__).resolve().parent.parent

PRE = REPO / "pre.md"
PUB = REPO / "pub.md"
PROS = REPO / "pros.md"
CATEGORY = REPO / "category.md"
SLIDES = REPO / "slides.md"

ListName = Literal["pre", "pub", "pros"]
LIST_LABELS: dict[ListName, str] = {
    "pre": "Preprints",
    "pub": "Publications",
    "pros": "Proceedings",
}
LIST_PATHS: dict[ListName, Path] = {"pre": PRE, "pub": PUB, "pros": PROS}

# Categories observed in the current category.md (order preserved on rebuild).
DEFAULT_CATEGORIES = [
    "Analysis",
    "Mathematical Physics and Modelling",
    "Numerical Analysis",
    "Scientific Computing",
]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

TITLE_RE = re.compile(r"^(\s*)(\d+)\.\s+_(.+?)_(\s*)$")
YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
ARXIV_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([\w.\-/]+?)(?:v\d+)?(?:\.pdf)?(?:[?#].*)?$", re.I)
DOI_RE = re.compile(r"doi\.org/(10\.[^\s]+)", re.I)
MORE = "<!--more-->"


def extract_links(links_line: str) -> list[tuple[str, str]]:
    """Return (label, url) pairs from an entry's links line."""
    return LINK_RE.findall(links_line)


def derive_doi(links: list[tuple[str, str]]) -> str | None:
    """Pick a DOI from the links; fall back to arXiv's DOI shim if available."""
    for _, url in links:
        m = DOI_RE.search(url)
        if m:
            return m.group(1).rstrip(".,;")
    for _, url in links:
        m = ARXIV_RE.search(url)
        if m:
            return f"10.48550/arXiv.{m.group(1)}"
    return None


def fetch_bibtex(doi: str, *, timeout: float = 15.0) -> str:
    """Fetch a BibTeX entry via DOI content negotiation (CrossRef / DataCite)."""
    import urllib.error
    import urllib.request

    req = urllib.request.Request(
        f"https://doi.org/{doi}",
        headers={
            "Accept": "application/x-bibtex; charset=utf-8",
            "User-Agent": "uzerbinati.eu-manager",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace").strip()


@dataclass
class Entry:
    """A paper / proceeding / preprint entry."""

    title: str
    authors_line: str  # raw, includes original leading whitespace
    links_line: str    # raw, includes year and any trailing whitespace
    year: int
    list_name: ListName
    category: str | None = None
    title_trailing_ws: str = ""  # any trailing spaces after the closing _

    def display_title(self) -> str:
        return self.title

    def title_line(self, position: int) -> str:
        return f"  {position}. _{self.title}_{self.title_trailing_ws}"


@dataclass
class ListFile:
    name: ListName
    path: Path
    header: str
    footer: str
    entries: list[Entry]
    more_after: int | None  # 1-indexed entry position the marker sits after

    def render(self) -> str:
        out = [self.header]
        for i, e in enumerate(self.entries, 1):
            out.append(e.title_line(i))
            out.append(e.authors_line)
            out.append(e.links_line)
            if self.more_after is not None and i == self.more_after:
                out.append(MORE)
        if self.footer:
            out.append(self.footer)
        return "\n".join(out)


def parse_list_file(name: ListName, path: Path) -> ListFile:
    text = path.read_text()
    lines = text.split("\n")

    start = next((i for i, l in enumerate(lines) if TITLE_RE.match(l)), None)
    if start is None:
        return ListFile(name, path, header=text, footer="", entries=[], more_after=None)

    header = "\n".join(lines[:start])
    entries: list[Entry] = []
    more_after: int | None = None

    i = start
    while i < len(lines):
        line = lines[i]
        if TITLE_RE.match(line):
            if i + 2 >= len(lines):
                break
            m = TITLE_RE.match(line)
            assert m
            title = m.group(3)
            trailing_ws = m.group(4)
            authors_line = lines[i + 1]
            links_line = lines[i + 2]
            year_match = list(YEAR_RE.finditer(links_line))
            year = int(year_match[-1].group(1)) if year_match else 0
            entries.append(
                Entry(
                    title=title,
                    authors_line=authors_line,
                    links_line=links_line,
                    year=year,
                    list_name=name,
                    title_trailing_ws=trailing_ws,
                )
            )
            i += 3
        elif line.strip() == MORE:
            more_after = len(entries)
            i += 1
        else:
            break

    footer = "\n".join(lines[i:])
    return ListFile(name, path, header=header, footer=footer, entries=entries, more_after=more_after)


def parse_category_index(path: Path) -> dict[str, str]:
    """Return {normalized_title -> category} from category.md."""
    text = path.read_text()
    index: dict[str, str] = {}
    current: str | None = None
    for line in text.split("\n"):
        if line.startswith("### "):
            current = line[4:].strip()
        elif current and (m := TITLE_RE.match(line)):
            index[normalize_title(m.group(3))] = current
    return index


def parse_category_categories(path: Path) -> list[str]:
    """Return the ordered list of `### …` headers currently used."""
    out: list[str] = []
    for line in path.read_text().split("\n"):
        if line.startswith("### "):
            name = line[4:].strip()
            if name not in out:
                out.append(name)
    return out or list(DEFAULT_CATEGORIES)


def normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", title.strip().lower())


# ---------------------------------------------------------------------------
# Talks (slides.md)
# ---------------------------------------------------------------------------

YEAR_HDR_RE = re.compile(r"^###\s+(\d{4})\s*$")
TALK_BULLET_RE = re.compile(r"^\s{2}\*\s")  # "  * "


@dataclass
class Talk:
    raw_block: str  # full multi-line bullet, no trailing newline


@dataclass
class YearSection:
    year: int
    talks: list[Talk]
    more_after: int | None  # 1-indexed talk after which <!--more--> sits
    trailing_blank: str = ""  # the blank line emitted between this section and the next


@dataclass
class TalksFile:
    path: Path
    header: str
    sections: list[YearSection]
    footer: str

    def render(self) -> str:
        out = [self.header]
        for sec in self.sections:
            out.append(f"### {sec.year}")
            for i, t in enumerate(sec.talks, 1):
                out.append(t.raw_block)
                if sec.more_after is not None and i == sec.more_after:
                    out.append(MORE)
            out.append(sec.trailing_blank)  # original blank-line separator (may have whitespace)
        # Drop the trailing blank we just appended if footer is empty,
        # to mimic the original file ending.
        if self.footer:
            out.append(self.footer)
        else:
            out.pop()
        return "\n".join(out)


def parse_talks_file(path: Path) -> TalksFile:
    text = path.read_text()
    lines = text.split("\n")

    start = next((i for i, l in enumerate(lines) if YEAR_HDR_RE.match(l)), None)
    if start is None:
        return TalksFile(path, header=text, sections=[], footer="")

    header = "\n".join(lines[:start])
    sections: list[YearSection] = []
    i = start
    cur: YearSection | None = None
    cur_talk_lines: list[str] | None = None

    def flush_talk():
        nonlocal cur_talk_lines
        if cur_talk_lines is not None and cur is not None:
            cur.talks.append(Talk(raw_block="\n".join(cur_talk_lines)))
            cur_talk_lines = None

    while i < len(lines):
        line = lines[i]
        if (m := YEAR_HDR_RE.match(line)):
            flush_talk()
            cur = YearSection(year=int(m.group(1)), talks=[], more_after=None)
            sections.append(cur)
            i += 1
            continue
        if cur is None:
            break
        if TALK_BULLET_RE.match(line):
            flush_talk()
            cur_talk_lines = [line]
            i += 1
            continue
        if line.strip() == MORE:
            flush_talk()
            cur.more_after = len(cur.talks)
            i += 1
            continue
        if line.strip() == "":
            # Look ahead: if the next non-blank line opens a new section, this
            # line is the section trailing separator (may carry whitespace).
            j = i + 1
            while j < len(lines) and lines[j].strip() == "":
                j += 1
            if cur is not None and j < len(lines) and YEAR_HDR_RE.match(lines[j]):
                flush_talk()
                cur.trailing_blank = line
                i = j
                continue
            # Otherwise: just a talk separator within a section.
            flush_talk()
            i += 1
            continue
        if cur_talk_lines is not None:
            # Continuation line of the current talk; preserve trailing whitespace.
            cur_talk_lines.append(line)
            i += 1
            continue
        # Unrecognized line outside a talk — treat as footer start.
        break

    flush_talk()
    footer = "\n".join(lines[i:])
    return TalksFile(path=path, header=header, sections=sections, footer=footer)


def render_new_talk(title: str, dates: str, location: str, extras: str) -> str:
    """Build a fresh talk bullet from form fields."""
    extras = extras.strip()
    suffix = f" {extras}" if extras else ""
    return f"  * {title}\n    {dates}\n    {location}{suffix}".rstrip()


# ---------------------------------------------------------------------------
# category.md rebuild
# ---------------------------------------------------------------------------

CATEGORY_HEADER = """---
layout: page
title:  Works by Category
---
A list of papers, proceedings, and preprints by category.
"""


def rebuild_category(
    categories: list[str],
    all_entries: list[Entry],
) -> str:
    """Emit category.md from the union of entries assigned to each category."""
    out = [CATEGORY_HEADER.rstrip()]
    seen_cats = list(categories)
    for e in all_entries:
        if e.category and e.category not in seen_cats:
            seen_cats.append(e.category)

    for cat in seen_cats:
        in_cat = [e for e in all_entries if (e.category or "") == cat]
        if not in_cat:
            continue
        # Stable sort by year desc — within a year, preserve source-list order.
        in_cat.sort(key=lambda e: -e.year)
        out.append(f"### {cat}")
        out.append("")
        for i, e in enumerate(in_cat, 1):
            out.append(e.title_line(i))
            out.append(e.authors_line)
            out.append(e.links_line)
        out.append("")
    return "\n".join(out).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------


@dataclass
class Document:
    pre: ListFile
    pub: ListFile
    pros: ListFile
    talks: TalksFile
    categories: list[str]
    dirty: set[str] = field(default_factory=set)

    @classmethod
    def load(cls) -> "Document":
        pre = parse_list_file("pre", PRE)
        pub = parse_list_file("pub", PUB)
        pros = parse_list_file("pros", PROS)
        talks = parse_talks_file(SLIDES)
        cat_index = parse_category_index(CATEGORY)
        cats = parse_category_categories(CATEGORY)
        # Stamp categories onto entries by title lookup.
        for lf in (pre, pub, pros):
            for e in lf.entries:
                e.category = cat_index.get(normalize_title(e.title))
        return cls(pre=pre, pub=pub, pros=pros, talks=talks, categories=cats)

    def list_for(self, name: ListName) -> ListFile:
        return {"pre": self.pre, "pub": self.pub, "pros": self.pros}[name]

    def all_entries(self) -> list[Entry]:
        return [*self.pre.entries, *self.pub.entries, *self.pros.entries]

    def mark(self, *names: str) -> None:
        for n in names:
            self.dirty.add(n)

    def move_entry(self, entry: Entry, target: ListName) -> Entry:
        src = self.list_for(entry.list_name)
        dst = self.list_for(target)
        src.entries.remove(entry)
        entry.list_name = target
        dst.entries.insert(0, entry)
        self.mark(src.name, dst.name, "category")
        return entry

    def add_entry(self, entry: Entry) -> None:
        self.list_for(entry.list_name).entries.insert(0, entry)
        self.mark(entry.list_name, "category")

    def delete_entry(self, entry: Entry) -> None:
        self.list_for(entry.list_name).entries.remove(entry)
        self.mark(entry.list_name, "category")

    def update_entry(self, entry: Entry) -> None:
        self.mark(entry.list_name, "category")

    def add_talk(self, year: int, talk: Talk) -> None:
        sec = next((s for s in self.talks.sections if s.year == year), None)
        if sec is None:
            sec = YearSection(year=year, talks=[], more_after=None)
            self.talks.sections.append(sec)
            self.talks.sections.sort(key=lambda s: -s.year)
        sec.talks.insert(0, talk)
        self.mark("slides")

    def delete_talk(self, year: int, talk: Talk) -> None:
        sec = next((s for s in self.talks.sections if s.year == year), None)
        if sec is None:
            return
        sec.talks.remove(talk)
        if not sec.talks:
            self.talks.sections.remove(sec)
        self.mark("slides")

    def save(self) -> list[Path]:
        written: list[Path] = []
        if "pre" in self.dirty:
            self._write(PRE, self.pre.render())
            written.append(PRE)
        if "pub" in self.dirty:
            self._write(PUB, self.pub.render())
            written.append(PUB)
        if "pros" in self.dirty:
            self._write(PROS, self.pros.render())
            written.append(PROS)
        if "category" in self.dirty:
            self._write(CATEGORY, rebuild_category(self.categories, self.all_entries()))
            written.append(CATEGORY)
        if "slides" in self.dirty:
            self._write(SLIDES, self.talks.render())
            written.append(SLIDES)
        self.dirty.clear()
        return written

    @staticmethod
    def _write(path: Path, content: str) -> None:
        # Preserve presence/absence of trailing newline of the original file.
        original = path.read_text()
        if original.endswith("\n") and not content.endswith("\n"):
            content += "\n"
        elif not original.endswith("\n") and content.endswith("\n"):
            content = content.rstrip("\n")
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(content)
        tmp.replace(path)


# ---------------------------------------------------------------------------
# Headless modes
# ---------------------------------------------------------------------------


def run_check() -> int:
    """Round-trip parse + re-emit. Diff against original; fail if non-empty."""
    rc = 0
    for name, path in [
        ("pre", PRE),
        ("pub", PUB),
        ("pros", PROS),
        ("slides", SLIDES),
    ]:
        original = path.read_text()
        if name == "slides":
            rendered = parse_talks_file(path).render()
        else:
            rendered = parse_list_file(name, path).render()  # type: ignore[arg-type]
        # Account for trailing newline policy in writer.
        if original.endswith("\n") and not rendered.endswith("\n"):
            rendered += "\n"
        if rendered != original:
            rc = 1
            print(f"=== {path.name} round-trip diff ===", file=sys.stderr)
            diff = difflib.unified_diff(
                original.splitlines(keepends=True),
                rendered.splitlines(keepends=True),
                fromfile=f"{path.name} (original)",
                tofile=f"{path.name} (rendered)",
            )
            sys.stderr.writelines(diff)
    if rc == 0:
        print("Round-trip OK for pre/pub/pros/slides.")
    return rc


def run_rebuild() -> int:
    doc = Document.load()
    doc.mark("category")
    written = doc.save()
    for p in written:
        print(f"wrote {p.relative_to(REPO)}")
    return 0


# ---------------------------------------------------------------------------
# Textual UI
# ---------------------------------------------------------------------------


def run_tui() -> int:  # pragma: no cover - interactive
    import webbrowser
    from datetime import datetime

    from rich.text import Text
    from textual import work
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Container, Horizontal, Vertical
    from textual.screen import ModalScreen
    from textual.widgets import (
        Button,
        DataTable,
        Footer,
        Header,
        Input,
        Label,
        Static,
        TabbedContent,
        TabPane,
        TextArea,
    )

    import scholar  # tools/scholar.py — sibling module

    SELF_COLOR = "orange3"       # with self-citations (Scholar)
    NON_SELF_COLOR = "green3"    # without self-citations (OpenAlex)

    LIST_TABS: list[ListName] = ["pre", "pub", "pros"]

    class EntryForm(ModalScreen):
        """Modal for add / edit. Returns updated dict or None."""

        DEFAULT_CSS = """
        EntryForm { align: center middle; }
        #form { width: 90; height: auto; padding: 1 2; background: $surface; border: solid $primary; }
        #form Input, #form Select { margin-bottom: 1; }
        #buttons { height: 3; align-horizontal: right; }
        Label.heading { text-style: bold; padding-bottom: 1; }
        """

        BINDINGS = [
            Binding("escape", "cancel", "Cancel"),
            Binding("ctrl+s", "save", "Save"),
        ]

        def __init__(self, entry: Entry | None, categories: list[str], heading: str):
            super().__init__()
            self.entry = entry
            self.categories = categories
            self.heading = heading

        def action_cancel(self) -> None:
            self.dismiss(None)

        def action_save(self) -> None:
            self._submit()

        def _submit(self) -> None:
            try:
                year = int(self.query_one("#year", Input).value.strip())
            except ValueError:
                self.app.notify("Year must be an integer.", severity="error")
                return
            title = self.query_one("#title", Input).value.strip()
            authors = self.query_one("#authors", Input).value.strip()
            links = self.query_one("#links", Input).value.strip()
            cat_raw = self.query_one("#category", Input).value.strip()
            cat_value = cat_raw or None
            if not title or not authors or not links:
                self.app.notify("Title, authors and links are required.", severity="error")
                return
            self.dismiss(
                {
                    "title": title,
                    "authors": authors,
                    "links": links,
                    "year": year,
                    "category": cat_value,
                }
            )

        def compose(self) -> ComposeResult:
            e = self.entry
            cat_value = e.category if e and e.category in self.categories else ""
            yield Container(
                Label(self.heading, classes="heading"),
                Input(value=(e.title if e else ""), placeholder="Title", id="title"),
                Input(
                    value=(e.authors_line.strip() if e else ""),
                    placeholder="Authors (e.g. Patrick E. Farrell, UZ)",
                    id="authors",
                ),
                Input(
                    value=(self._strip_links_year(e) if e else ""),
                    placeholder="Links (e.g. [arXiv](https://...), [JOSS](https://...))",
                    id="links",
                ),
                Input(
                    value=(str(e.year) if e else ""),
                    placeholder="Year (YYYY)",
                    id="year",
                ),
                Input(
                    value=cat_value,
                    placeholder=f"Category — one of: {', '.join(self.categories)}",
                    id="category",
                ),
                Horizontal(
                    Button("Save", id="save", variant="primary"),
                    Button("Cancel", id="cancel"),
                    id="buttons",
                ),
                id="form",
            )

        @staticmethod
        def _strip_links_year(e: Entry) -> str:
            # Strip the trailing ", YYYY." (and optional trailing whitespace)
            line = e.links_line.strip()
            return re.sub(r",\s*\d{4}\.?\s*$", "", line)

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "cancel":
                self.dismiss(None)
            else:
                self._submit()

    class MoveModal(ModalScreen):
        DEFAULT_CSS = """
        MoveModal { align: center middle; }
        #panel { width: 70; height: auto; padding: 1 2; background: $surface; border: solid $primary; }
        #buttons { height: 3; align-horizontal: right; }
        Button.target { width: 1fr; margin-right: 1; }
        """

        # Letter accelerators: first character of the list label.
        BINDINGS = [
            Binding("escape", "cancel", "Cancel"),
            Binding("p", "pick('pre')", "Preprints", show=False),
            Binding("u", "pick('pub')", "Publications", show=False),
            Binding("r", "pick('pros')", "Proceedings", show=False),
        ]

        def __init__(self, current: ListName):
            super().__init__()
            self.current = current

        def compose(self) -> ComposeResult:
            buttons = [
                Button(
                    f"{LIST_LABELS[n]}  [{self._key_for(n)}]",
                    id=f"to-{n}",
                    classes="target",
                )
                for n in LIST_TABS
                if n != self.current
            ]
            yield Container(
                Label(
                    f"Move to (currently: {LIST_LABELS[self.current]})  —  press p/u/r or Esc",
                    classes="heading",
                ),
                Horizontal(*buttons),
                Horizontal(Button("Cancel", id="cancel"), id="buttons"),
                id="panel",
            )

        @staticmethod
        def _key_for(name: ListName) -> str:
            return {"pre": "p", "pub": "u", "pros": "r"}[name]

        def action_cancel(self) -> None:
            self.dismiss(None)

        def action_pick(self, name: ListName) -> None:
            if name != self.current:
                self.dismiss(name)

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "cancel":
                self.dismiss(None)
                return
            if event.button.id and event.button.id.startswith("to-"):
                self.dismiss(event.button.id[3:])

    class ConfirmModal(ModalScreen):
        DEFAULT_CSS = """
        ConfirmModal { align: center middle; }
        #panel { width: 60; height: auto; padding: 1 2; background: $surface; border: solid $error; }
        #buttons { height: 3; align-horizontal: right; }
        """

        BINDINGS = [
            Binding("escape", "cancel", "Cancel"),
            Binding("y", "confirm", "Yes"),
            Binding("enter", "confirm", "Yes"),
            Binding("n", "cancel", "No"),
        ]

        def __init__(self, message: str):
            super().__init__()
            self.message = message

        def compose(self) -> ComposeResult:
            yield Container(
                Label(f"{self.message}  —  y to confirm, n/Esc to cancel", classes="heading"),
                Horizontal(
                    Button("Delete (y)", id="ok", variant="error"),
                    Button("Cancel (n)", id="cancel"),
                    id="buttons",
                ),
                id="panel",
            )

        def action_confirm(self) -> None:
            self.dismiss(True)

        def action_cancel(self) -> None:
            self.dismiss(False)

        def on_button_pressed(self, event: Button.Pressed) -> None:
            self.dismiss(event.button.id == "ok")

    class TalkEditForm(ModalScreen):
        """Free-form raw-text edit of a talk block. Year is editable and may
        move the talk into another year section."""

        DEFAULT_CSS = """
        TalkEditForm { align: center middle; }
        #form { width: 110; height: auto; padding: 1 2; background: $surface; border: solid $primary; }
        #editor { height: 14; margin-bottom: 1; }
        Input { margin-bottom: 1; }
        #buttons { height: 3; align-horizontal: right; }
        """

        BINDINGS = [
            Binding("escape", "cancel", "Cancel"),
            Binding("ctrl+s", "save", "Save"),
        ]

        def __init__(self, raw: str, year: int):
            super().__init__()
            self.raw = raw
            self.year = year

        def compose(self) -> ComposeResult:
            yield Container(
                Label(
                    "Edit talk — keep the leading '  * ' bullet and 4-space "
                    "continuation indent. Change Year to move between sections.  "
                    "(Ctrl+S save, Esc cancel)",
                    classes="heading",
                ),
                Input(value=str(self.year), placeholder="Year (YYYY)", id="year"),
                TextArea(text=self.raw, id="editor"),
                Horizontal(
                    Button("Save", id="save", variant="primary"),
                    Button("Cancel", id="cancel"),
                    id="buttons",
                ),
                id="form",
            )

        def action_cancel(self) -> None:
            self.dismiss(None)

        def action_save(self) -> None:
            self._submit()

        def _submit(self) -> None:
            try:
                year = int(self.query_one("#year", Input).value.strip())
            except ValueError:
                self.app.notify("Year must be an integer.", severity="error")
                return
            text = self.query_one("#editor", TextArea).text.rstrip("\n")
            if not text.strip():
                self.app.notify("Talk body cannot be empty.", severity="error")
                return
            self.dismiss({"raw_block": text, "year": year})

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "cancel":
                self.dismiss(None)
            else:
                self._submit()

    class TalkForm(ModalScreen):
        DEFAULT_CSS = """
        TalkForm { align: center middle; }
        #form { width: 90; height: auto; padding: 1 2; background: $surface; border: solid $primary; }
        #form Input { margin-bottom: 1; }
        #buttons { height: 3; align-horizontal: right; }
        """

        BINDINGS = [
            Binding("escape", "cancel", "Cancel"),
            Binding("ctrl+s", "save", "Save"),
        ]

        def compose(self) -> ComposeResult:
            yield Container(
                Label("Add talk / visit  (Ctrl+S save, Esc cancel)", classes="heading"),
                Input(placeholder="Title (e.g. ENUMATH)", id="title"),
                Input(placeholder="Date(s) (e.g. 14/09 - 19/09)", id="dates"),
                Input(placeholder="Location (e.g. Heidelberg, DE)", id="location"),
                Input(placeholder="Extras (e.g. [slides](https://…))", id="extras"),
                Input(placeholder="Year (YYYY)", id="year"),
                Horizontal(
                    Button("Save", id="save", variant="primary"),
                    Button("Cancel", id="cancel"),
                    id="buttons",
                ),
                id="form",
            )

        def action_cancel(self) -> None:
            self.dismiss(None)

        def action_save(self) -> None:
            self._submit()

        def _submit(self) -> None:
            try:
                year = int(self.query_one("#year", Input).value.strip())
            except ValueError:
                self.app.notify("Year must be an integer.", severity="error")
                return
            title = self.query_one("#title", Input).value.strip()
            dates = self.query_one("#dates", Input).value.strip()
            location = self.query_one("#location", Input).value.strip()
            extras = self.query_one("#extras", Input).value.strip()
            if not (title and dates and location):
                self.app.notify("Title, dates, location are required.", severity="error")
                return
            self.dismiss(
                {"year": year, "title": title, "dates": dates, "location": location, "extras": extras}
            )

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "cancel":
                self.dismiss(None)
            else:
                self._submit()

    class ScholarConfigForm(ModalScreen):
        DEFAULT_CSS = """
        ScholarConfigForm { align: center middle; }
        #form { width: 80; height: auto; padding: 1 2; background: $surface; border: solid $primary; }
        Input { margin-bottom: 1; }
        #buttons { height: 3; align-horizontal: right; }
        Label.heading { text-style: bold; padding-bottom: 1; }
        """

        BINDINGS = [
            Binding("escape", "cancel", "Cancel"),
            Binding("ctrl+s", "save", "Save"),
        ]

        def __init__(self, current: str = "") -> None:
            super().__init__()
            self.current = current

        def compose(self) -> ComposeResult:
            yield Container(
                Label(
                    "Google Scholar user id  —  the value of `user=…` in your "
                    "profile URL.  (Ctrl+S save, Esc cancel)",
                    classes="heading",
                ),
                Input(
                    value=self.current,
                    placeholder="e.g. bLUNjmgAAAAJ",
                    id="user_id",
                ),
                Horizontal(
                    Button("Save", id="save", variant="primary"),
                    Button("Cancel", id="cancel"),
                    id="buttons",
                ),
                id="form",
            )

        def action_cancel(self) -> None:
            self.dismiss(None)

        def action_save(self) -> None:
            self._submit()

        def _submit(self) -> None:
            uid = self.query_one("#user_id", Input).value.strip()
            if not uid:
                self.app.notify("user id cannot be empty.", severity="error")
                return
            self.dismiss(uid)

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "cancel":
                self.dismiss(None)
            else:
                self._submit()

    class PaperInfoModal(ModalScreen):
        """Quick-share modal — copy links and BibTeX for an entry to the clipboard."""

        DEFAULT_CSS = """
        PaperInfoModal { align: center middle; }
        #panel { width: 120; height: auto; padding: 1 2; background: $surface; border: solid $primary; }
        #title { text-style: bold; padding-bottom: 1; }
        #meta, #links { margin-bottom: 1; }
        #bibtex { padding: 1; background: $boost; }
        """

        BINDINGS = [
            Binding("escape", "cancel", "Close"),
            Binding("q", "cancel", "Close", show=False),
            Binding("b", "copy_bibtex", "Copy BibTeX"),
            Binding("1", "copy_link(0)", show=False),
            Binding("2", "copy_link(1)", show=False),
            Binding("3", "copy_link(2)", show=False),
            Binding("4", "copy_link(3)", show=False),
            Binding("5", "copy_link(4)", show=False),
            Binding("6", "copy_link(5)", show=False),
            Binding("7", "copy_link(6)", show=False),
            Binding("8", "copy_link(7)", show=False),
            Binding("9", "copy_link(8)", show=False),
        ]

        def __init__(self, entry: Entry):
            super().__init__()
            self.entry = entry
            self.links = extract_links(entry.links_line)
            self.doi = derive_doi(self.links)
            self.bibtex: str | None = None

        def compose(self) -> ComposeResult:
            cat = f"  ·  {self.entry.category}" if self.entry.category else ""
            meta = (
                f"{self.entry.authors_line.strip()}\n"
                f"{LIST_LABELS[self.entry.list_name]}  ·  {self.entry.year}{cat}"
            )
            link_rows = []
            for i, (label, url) in enumerate(self.links, 1):
                link_rows.append(f"  [b][{i}][/b]  [b]{label}[/b]   [dim]{url}[/dim]")
            doi_hint = self.doi or "[red]no DOI / arXiv id found[/red]"
            link_rows.append(f"  [b][b][/b]  [b]BibTeX[/b]   [dim]via {doi_hint}[/dim]")
            link_rows.append("")
            link_rows.append(
                "[dim]press a number to copy that link, b for BibTeX, Esc to close[/dim]"
            )
            yield Container(
                Static(self.entry.title, id="title"),
                Static(meta, id="meta"),
                Static("\n".join(link_rows), id="links"),
                Static("", id="bibtex"),
                id="panel",
            )

        def action_cancel(self) -> None:
            self.dismiss(None)

        def _copy(self, text: str) -> None:
            try:
                self.app.copy_to_clipboard(text)
            except Exception:
                # Older Textual fallback: print to status; user can re-copy from terminal selection
                pass

        def action_copy_link(self, idx: int) -> None:
            if 0 <= idx < len(self.links):
                label, url = self.links[idx]
                self._copy(url)
                self.app.notify(f"copied {label}: {url}", timeout=4)

        def action_copy_bibtex(self) -> None:
            if self.bibtex:
                self._copy(self.bibtex)
                self.app.notify("copied BibTeX (cached)", timeout=4)
                return
            if not self.doi:
                self.app.notify(
                    "no DOI or arXiv id found in this entry's links",
                    severity="warning",
                )
                return
            self.app.notify(f"fetching BibTeX for {self.doi}…")
            self._fetch_bibtex_worker()

        @work(thread=True, exclusive=True, group="bibtex")
        def _fetch_bibtex_worker(self) -> None:
            assert self.doi is not None
            app = self.app
            try:
                bib = fetch_bibtex(self.doi)
            except Exception as e:
                app.call_from_thread(
                    app.notify, f"BibTeX fetch failed: {e}", severity="error"
                )
                return
            self.bibtex = bib
            app.call_from_thread(self._copy, bib)
            app.call_from_thread(self.query_one("#bibtex", Static).update, bib)
            app.call_from_thread(app.notify, "copied BibTeX")

    class WebsiteApp(App):
        TITLE = "uzerbinati.eu manager"
        CSS = """
        Screen { layout: vertical; }
        DataTable { height: 1fr; }
        #status { dock: bottom; height: 1; padding: 0 1; background: $boost; }
        #scholar-header { height: auto; padding: 0 1; background: $boost; }
        #scholar-subtabs { height: 1fr; }
        #scholar-trends { height: 1fr; padding: 1 2; }
        """
        BINDINGS = [
            Binding("a", "add", "Add"),
            Binding("e", "edit", "Edit"),
            Binding("m", "move", "Move"),
            Binding("d", "delete", "Delete"),
            Binding("i", "info", "Info / share"),
            Binding("r", "rebuild", "Rebuild category"),
            Binding("s", "save", "Save"),
            Binding("q", "quit_safe", "Quit"),
            Binding("R", "refresh_scholar", "Refresh scholar"),
            Binding("c", "configure_scholar", "Configure scholar"),
            Binding("1", "tab('pre')", "Preprints", show=False),
            Binding("2", "tab('pub')", "Publications", show=False),
            Binding("3", "tab('pros')", "Proceedings", show=False),
            Binding("4", "tab('talks')", "Talks", show=False),
            Binding("5", "tab('scholar')", "Scholar", show=False),
        ]

        def __init__(self) -> None:
            super().__init__()
            self.doc = Document.load()
            self.scholar_profile: scholar.Profile | None = None
            self.scholar_busy: bool = False

        def compose(self) -> ComposeResult:
            yield Header()
            with TabbedContent(initial="tab-pre", id="main-tabs"):
                with TabPane(LIST_LABELS["pre"], id="tab-pre"):
                    yield DataTable(id="table-pre", cursor_type="row", zebra_stripes=True)
                with TabPane(LIST_LABELS["pub"], id="tab-pub"):
                    yield DataTable(id="table-pub", cursor_type="row", zebra_stripes=True)
                with TabPane(LIST_LABELS["pros"], id="tab-pros"):
                    yield DataTable(id="table-pros", cursor_type="row", zebra_stripes=True)
                with TabPane("Talks", id="tab-talks"):
                    yield DataTable(id="table-talks", cursor_type="row", zebra_stripes=True)
                with TabPane("Scholar", id="tab-scholar"):
                    yield Static("no cached snapshot — press c to configure", id="scholar-header")
                    with TabbedContent(id="scholar-subtabs", initial="sub-papers"):
                        with TabPane("Papers", id="sub-papers"):
                            yield DataTable(
                                id="table-scholar-papers",
                                cursor_type="row",
                                zebra_stripes=True,
                            )
                        with TabPane("Recent Citations", id="sub-recent"):
                            yield DataTable(
                                id="table-scholar-recent",
                                cursor_type="row",
                                zebra_stripes=True,
                            )
                        with TabPane("Per-year", id="sub-per-year"):
                            yield Static("no per-year data", id="scholar-trends")
            yield Static("ready", id="status")
            yield Footer()

        def on_mount(self) -> None:
            for n in LIST_TABS:
                t = self.query_one(f"#table-{n}", DataTable)
                t.add_columns("#", "Year", "Title", "Category")
            tt = self.query_one("#table-talks", DataTable)
            tt.add_columns("Year", "Title (first line)")
            self.query_one("#table-scholar-papers", DataTable).add_columns(
                "#",
                "Year",
                Text("Cited by", style=SELF_COLOR),
                Text("Non-self", style=NON_SELF_COLOR),
                "Title",
                "Venue",
            )
            citing_hdr = Text()
            citing_hdr.append("Citing paper  ")
            citing_hdr.append("(", style="dim")
            citing_hdr.append("self", style=SELF_COLOR)
            citing_hdr.append(" / ", style="dim")
            citing_hdr.append("non-self", style=NON_SELF_COLOR)
            citing_hdr.append(")", style="dim")
            self.query_one("#table-scholar-recent", DataTable).add_columns(
                "Date", citing_hdr, "Authors", "Cites (of mine)"
            )
            self.refresh_all()

            # Stale-while-revalidate: paint cache immediately, then refresh in
            # the background if a user id is configured.
            cached = scholar.load_cache()
            if cached is not None:
                self.scholar_profile = cached
                self._render_scholar(cached)
            cfg = scholar.load_config()
            if cfg.get("user_id"):
                self._scholar_refresh_worker()
            elif cached is None:
                self.query_one("#scholar-header", Static).update(
                    "no Scholar user id configured — press [b]c[/b] to set one"
                )

            self.call_after_refresh(self._focus_active_table)

        def _focus_active_table(self) -> None:
            try:
                outer = self.query_one("#main-tabs", TabbedContent)
            except Exception:
                return
            tab = outer.active
            if tab == "tab-scholar":
                try:
                    sub = self.query_one("#scholar-subtabs", TabbedContent).active
                except Exception:
                    sub = "sub-papers"
                target_id = {
                    "sub-papers": "table-scholar-papers",
                    "sub-recent": "table-scholar-recent",
                    "sub-per-year": "scholar-trends",
                }.get(sub, "table-scholar-papers")
            else:
                target_id = tab.replace("tab-", "table-") if tab else "table-pre"
            try:
                self.query_one(f"#{target_id}", DataTable).focus()
            except Exception:
                pass

        def on_tabbed_content_tab_activated(self, event) -> None:
            self.call_after_refresh(self._focus_active_table)

        # --- helpers ---------------------------------------------------------

        def refresh_all(self) -> None:
            for n in LIST_TABS:
                self._refresh_list(n)
            self._refresh_talks()
            self._update_status()

        def _refresh_list(self, name: ListName) -> None:
            t = self.query_one(f"#table-{name}", DataTable)
            t.clear()
            for i, e in enumerate(self.doc.list_for(name).entries, 1):
                t.add_row(
                    str(i),
                    str(e.year),
                    self._truncate(e.title, 70),
                    e.category or "—",
                    key=str(id(e)),
                )

        def _refresh_talks(self) -> None:
            t = self.query_one("#table-talks", DataTable)
            t.clear()
            for sec in self.doc.talks.sections:
                for talk in sec.talks:
                    first = talk.raw_block.split("\n", 1)[0].lstrip("* ").lstrip()
                    t.add_row(str(sec.year), self._truncate(first, 90), key=str(id(talk)))

        @staticmethod
        def _truncate(s: str, n: int) -> str:
            return s if len(s) <= n else s[: n - 1] + "…"

        def _update_status(self) -> None:
            dirty = ", ".join(sorted(self.doc.dirty)) if self.doc.dirty else "none"
            self.query_one("#status", Static).update(f"unsaved: {dirty}")

        def _active_list(self) -> ListName | None:
            tabs = self.query_one("#main-tabs", TabbedContent)
            tab = tabs.active
            for n in LIST_TABS:
                if tab == f"tab-{n}":
                    return n
            return None

        def _selected_entry(self) -> Entry | None:
            n = self._active_list()
            if n is None:
                return None
            t = self.query_one(f"#table-{n}", DataTable)
            if t.cursor_row is None or t.cursor_row < 0:
                return None
            entries = self.doc.list_for(n).entries
            if t.cursor_row >= len(entries):
                return None
            return entries[t.cursor_row]

        def _selected_talk(self) -> tuple[YearSection, Talk] | None:
            t = self.query_one("#table-talks", DataTable)
            row = t.cursor_row
            if row is None or row < 0:
                return None
            counter = 0
            for sec in self.doc.talks.sections:
                for talk in sec.talks:
                    if counter == row:
                        return sec, talk
                    counter += 1
            return None

        # --- actions ---------------------------------------------------------

        def action_add(self) -> None:
            tab = self.query_one("#main-tabs", TabbedContent).active
            if tab == "tab-talks":
                self.push_screen(TalkForm(), self._on_talk_added)
                return
            name = self._active_list()
            if name is None:
                return
            self.push_screen(
                EntryForm(None, self.doc.categories, f"Add to {LIST_LABELS[name]}"),
                lambda res: self._on_entry_added(name, res),
            )

        def _on_entry_added(self, name: ListName, res: dict | None) -> None:
            if not res:
                return
            entry = build_entry(name, res)
            self.doc.add_entry(entry)
            self.refresh_all()

        def _on_talk_added(self, res: dict | None) -> None:
            if not res:
                return
            block = render_new_talk(res["title"], res["dates"], res["location"], res["extras"])
            self.doc.add_talk(res["year"], Talk(raw_block=block))
            self.refresh_all()

        def action_edit(self) -> None:
            tab = self.query_one("#main-tabs", TabbedContent).active
            if tab == "tab-talks":
                sel = self._selected_talk()
                if sel is None:
                    return
                sec, talk = sel
                self.push_screen(
                    TalkEditForm(talk.raw_block, sec.year),
                    lambda res: self._on_talk_edited(talk, sec, res),
                )
                return
            entry = self._selected_entry()
            if entry is None:
                return
            self.push_screen(
                EntryForm(entry, self.doc.categories, "Edit entry"),
                lambda res: self._on_entry_edited(entry, res),
            )

        def _on_entry_edited(self, entry: Entry, res: dict | None) -> None:
            if not res:
                return
            apply_form_to_entry(entry, res)
            self.doc.update_entry(entry)
            self.refresh_all()

        def _on_talk_edited(
            self, talk: Talk, sec: YearSection, res: dict | None
        ) -> None:
            if not res:
                return
            talk.raw_block = res["raw_block"]
            new_year = res["year"]
            if new_year != sec.year:
                sec.talks.remove(talk)
                if not sec.talks:
                    self.doc.talks.sections.remove(sec)
                target = next(
                    (s for s in self.doc.talks.sections if s.year == new_year), None
                )
                if target is None:
                    target = YearSection(year=new_year, talks=[], more_after=None)
                    self.doc.talks.sections.append(target)
                    self.doc.talks.sections.sort(key=lambda s: -s.year)
                target.talks.insert(0, talk)
            self.doc.mark("slides")
            self.refresh_all()

        def action_move(self) -> None:
            entry = self._selected_entry()
            if entry is None:
                return
            self.push_screen(
                MoveModal(entry.list_name),
                lambda target: self._on_moved(entry, target),
            )

        def _on_moved(self, entry: Entry, target: ListName | None) -> None:
            if not target:
                return
            self.doc.move_entry(entry, target)
            # Open edit form so user can append the venue link.
            self.push_screen(
                EntryForm(entry, self.doc.categories, f"Edit (just moved to {LIST_LABELS[target]})"),
                lambda res: self._on_entry_edited(entry, res),
            )

        def action_delete(self) -> None:
            tab = self.query_one("#main-tabs", TabbedContent).active
            if tab == "tab-talks":
                sel = self._selected_talk()
                if sel is None:
                    return
                sec, talk = sel
                self.push_screen(
                    ConfirmModal(f"Delete this talk from {sec.year}?"),
                    lambda ok: self._on_talk_deleted(sec.year, talk, ok),
                )
                return
            entry = self._selected_entry()
            if entry is None:
                return
            self.push_screen(
                ConfirmModal(f"Delete '{entry.title[:60]}'?"),
                lambda ok: self._on_entry_deleted(entry, ok),
            )

        def _on_entry_deleted(self, entry: Entry, ok: bool | None) -> None:
            if not ok:
                return
            self.doc.delete_entry(entry)
            self.refresh_all()

        def _on_talk_deleted(self, year: int, talk: Talk, ok: bool | None) -> None:
            if not ok:
                return
            self.doc.delete_talk(year, talk)
            self.refresh_all()

        def action_info(self) -> None:
            entry = self._selected_entry()
            if entry is None:
                return
            self.push_screen(PaperInfoModal(entry))

        def action_rebuild(self) -> None:
            self.doc.mark("category")
            self.notify("category.md will be rebuilt on save.")
            self._update_status()

        def action_save(self) -> None:
            written = self.doc.save()
            if not written:
                self.notify("Nothing to save.")
            else:
                self.notify("Saved: " + ", ".join(p.name for p in written))
            self.refresh_all()

        def action_tab(self, name: str) -> None:
            tabs = self.query_one("#main-tabs", TabbedContent)
            tabs.active = f"tab-{name}"

        def action_quit_safe(self) -> None:
            if self.doc.dirty:
                self.push_screen(
                    ConfirmModal("Discard unsaved changes and quit?"),
                    lambda ok: self.exit() if ok else None,
                )
            else:
                self.exit()

        # --- scholar --------------------------------------------------------

        def action_refresh_scholar(self) -> None:
            cfg = scholar.load_config()
            if not cfg.get("user_id"):
                self.notify("configure a Scholar user id first (press c)", severity="warning")
                return
            if self.scholar_busy:
                self.notify("a Scholar refresh is already running")
                return
            self._scholar_refresh_worker()

        def action_configure_scholar(self) -> None:
            cur = scholar.load_config().get("user_id", "")
            self.push_screen(ScholarConfigForm(cur), self._on_scholar_configured)

        def _on_scholar_configured(self, user_id: str | None) -> None:
            if not user_id:
                return
            scholar.save_config({"user_id": user_id})
            self.notify(f"saved Scholar user id: {user_id}")
            if not self.scholar_busy:
                self._scholar_refresh_worker()

        @work(thread=True, exclusive=True, group="scholar")
        def _scholar_refresh_worker(self) -> None:
            cfg = scholar.load_config()
            user_id = cfg.get("user_id")
            if not user_id:
                self.call_from_thread(
                    self.query_one("#status", Static).update,
                    "scholar: no user id configured",
                )
                return
            self.call_from_thread(self._set_scholar_busy, True)
            self.call_from_thread(
                self.query_one("#status", Static).update, "fetching scholar…"
            )
            previous = scholar.load_cache()
            try:
                profile = scholar.fetch_profile(
                    user_id,
                    polite_email=cfg.get("polite_email"),
                )
            except scholar.CaptchaError as e:
                # Scholar's IP-based rate limit kicked in on the profile page.
                # Nothing useful we can do client-side beyond waiting.
                if previous is not None:
                    self.call_from_thread(self._on_scholar_loaded, previous)
                else:
                    self.call_from_thread(self._set_scholar_busy, False)
                self.call_from_thread(
                    self.query_one("#status", Static).update,
                    "scholar: CAPTCHA — wait ~1 h, then Shift+R",
                )
                self.call_from_thread(
                    self.notify,
                    f"Scholar rate-limited the profile page ({e}). "
                    "Wait ~1 h and try Shift+R again.",
                    severity="warning",
                    timeout=20,
                )
                return
            except Exception as e:
                self.call_from_thread(
                    self.notify, f"scholar refresh failed: {e}", severity="error"
                )
                self.call_from_thread(
                    self.query_one("#status", Static).update,
                    "scholar refresh failed",
                )
                self.call_from_thread(self._set_scholar_busy, False)
                return
            # If the new fetch silently came back empty but we had citations
            # before, treat the previous list as authoritative — likely a
            # rate-limit we didn't classify as CAPTCHA.
            if (
                not profile.recent_citations
                and previous is not None
                and previous.recent_citations
            ):
                profile.recent_citations = previous.recent_citations
            try:
                scholar.save_cache(profile)
            except Exception as e:
                self.call_from_thread(
                    self.notify, f"scholar cache write failed: {e}", severity="warning"
                )
            self.call_from_thread(self._on_scholar_loaded, profile)

        def _set_scholar_busy(self, busy: bool) -> None:
            self.scholar_busy = busy

        def _on_scholar_loaded(self, profile: scholar.Profile) -> None:
            self.scholar_profile = profile
            self._render_scholar(profile)
            self._set_scholar_busy(False)
            self._update_status()
            self.notify(
                f"scholar refreshed: {profile.total_citations} cites, "
                f"h={profile.h_index}, {len(profile.recent_citations)} new citations"
            )

        def _render_scholar(self, p: scholar.Profile) -> None:
            self._render_scholar_header(p)
            self._render_scholar_papers(p)
            self._render_scholar_citations(p)
            self._render_scholar_trends(p)

        def _render_scholar_header(self, p: scholar.Profile) -> None:
            fetched = (
                datetime.fromtimestamp(p.fetched_at).strftime("%Y-%m-%d %H:%M")
                if p.fetched_at
                else "—"
            )
            interests = ", ".join(p.interests) if p.interests else "—"
            text = (
                f"[b]{p.name}[/b]   {p.affiliation}\n"
                f"interests: {interests}\n"
                f"[{SELF_COLOR}]with self (Scholar):   "
                f"[b]citations[/b] {p.total_citations} (5y {p.citations_5y})   "
                f"[b]h-index[/b] {p.h_index} (5y {p.h_index_5y})   "
                f"[b]i10[/b] {p.i10_index} (5y {p.i10_index_5y})[/{SELF_COLOR}]   "
                f"[dim]fetched {fetched}[/dim]"
            )
            if p.total_citations_non_self is not None:
                ns_5y = (
                    f" (5y {p.citations_5y_non_self})"
                    if p.citations_5y_non_self is not None
                    else ""
                )
                text += (
                    f"\n[{NON_SELF_COLOR}]non-self (OpenAlex):   "
                    f"[b]citations[/b] {p.total_citations_non_self}{ns_5y}   "
                    f"[b]h-index[/b] {p.h_index_non_self}   "
                    f"[b]i10[/b] {p.i10_index_non_self}[/{NON_SELF_COLOR}]"
                )
            self.query_one("#scholar-header", Static).update(text)

        def _render_scholar_trends(self, p: scholar.Profile) -> None:
            target = self.query_one("#scholar-trends", Static)
            if not p.citations_per_year:
                target.update(Text("no per-year data — refresh with Shift+R", style="dim"))
                return
            cpy = p.citations_per_year
            ns_cpy = p.citations_per_year_non_self or {}
            all_years = sorted(set(cpy) | set(ns_cpy), reverse=True)
            scale_max = max(
                [cpy.get(y, 0) for y in all_years] + [ns_cpy.get(y, 0) for y in all_years]
            ) or 1
            BAR_WIDTH = 50

            def bar(value: int) -> str:
                full = int(value * BAR_WIDTH / scale_max)
                remainder = (value * BAR_WIDTH / scale_max) - full
                eighth = " ▏▎▍▌▋▊▉"[min(7, int(remainder * 8))]
                return "█" * full + (eighth if eighth.strip() else "")

            total_self = sum(cpy.values())
            total_ns = sum(ns_cpy.values()) if ns_cpy else None

            out = Text()
            out.append("Citations per year", style="bold")
            out.append("   ")
            out.append(f"■ self (Scholar): {total_self}", style=SELF_COLOR)
            out.append("   ")
            if total_ns is not None:
                out.append(
                    f"■ non-self (OpenAlex): {total_ns}", style=NON_SELF_COLOR
                )
            else:
                out.append("■ non-self (OpenAlex): n/a", style="dim")
            out.append("\n\n")

            for y in all_years:
                sv = cpy.get(y, 0)
                out.append(f"  {y}  ", style="bold")
                out.append(
                    f"self      {bar(sv).ljust(BAR_WIDTH + 1)} {sv}",
                    style=SELF_COLOR,
                )
                out.append("\n")

                nv = ns_cpy.get(y)
                if nv is None:
                    out.append(
                        "        non-self  "
                        f"{'—'.ljust(BAR_WIDTH + 1)} (not on OpenAlex)",
                        style=f"{NON_SELF_COLOR} dim",
                    )
                else:
                    out.append(
                        "        non-self  "
                        f"{bar(nv).ljust(BAR_WIDTH + 1)} {nv}",
                        style=NON_SELF_COLOR,
                    )
                out.append("\n\n")

            target.update(out)

        def _render_scholar_papers(self, p: scholar.Profile) -> None:
            t = self.query_one("#table-scholar-papers", DataTable)
            t.clear()
            for i, paper in enumerate(p.papers, 1):
                if paper.cited_by_non_self is None:
                    ns_cell = Text("—", style="dim")
                else:
                    ns_cell = Text(str(paper.cited_by_non_self), style=NON_SELF_COLOR)
                t.add_row(
                    str(i),
                    str(paper.year) if paper.year else "—",
                    Text(str(paper.cited_by), style=SELF_COLOR),
                    ns_cell,
                    self._truncate(paper.title, 70),
                    self._truncate(paper.venue, 40),
                    key=f"paper-{i}",
                )

        def _render_scholar_citations(self, p: scholar.Profile) -> None:
            t = self.query_one("#table-scholar-recent", DataTable)
            t.clear()
            for i, c in enumerate(p.recent_citations, 1):
                if c.date_iso:
                    date_cell = c.date_iso
                elif c.year:
                    date_cell = str(c.year)
                else:
                    date_cell = "—"
                color = SELF_COLOR if c.is_self_citation else NON_SELF_COLOR
                t.add_row(
                    date_cell,
                    Text(self._truncate(c.title, 60), style=color),
                    self._truncate(c.authors, 35),
                    self._truncate(c.of_paper_title, 40),
                    key=f"cite-{i}",
                )

        def on_data_table_row_selected(self, event) -> None:
            tid = event.data_table.id
            if tid == "table-scholar-papers" and self.scholar_profile:
                row = event.cursor_row
                if 0 <= row < len(self.scholar_profile.papers):
                    url = self.scholar_profile.papers[row].paper_url
                    if url:
                        webbrowser.open(url)
            elif tid == "table-scholar-recent" and self.scholar_profile:
                row = event.cursor_row
                if 0 <= row < len(self.scholar_profile.recent_citations):
                    url = self.scholar_profile.recent_citations[row].citing_paper_url
                    if url:
                        webbrowser.open(url)
            elif tid in {"table-pre", "table-pub", "table-pros"}:
                entry = self._selected_entry()
                if entry is not None:
                    self.push_screen(PaperInfoModal(entry))

    WebsiteApp().run()
    return 0


# ---------------------------------------------------------------------------
# Helpers used by both UI and headless paths
# ---------------------------------------------------------------------------


def build_entry(list_name: ListName, fields: dict) -> Entry:
    title = fields["title"].strip()
    authors = fields["authors"].strip()
    links = fields["links"].strip()
    year = int(fields["year"])
    return Entry(
        title=title,
        authors_line=f"  {authors}",
        links_line=f"  {links}, {year}.",
        year=year,
        list_name=list_name,
        category=fields.get("category"),
    )


def apply_form_to_entry(entry: Entry, fields: dict) -> None:
    entry.title = fields["title"].strip()
    entry.authors_line = f"  {fields['authors'].strip()}"
    entry.links_line = f"  {fields['links'].strip()}, {int(fields['year'])}."
    entry.year = int(fields["year"])
    entry.category = fields.get("category")
    entry.title_trailing_ws = ""  # normalise on edit


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="uzerbinati.eu site manager")
    parser.add_argument("--check", action="store_true", help="parse + re-emit, fail on diff")
    parser.add_argument("--rebuild", action="store_true", help="rebuild category.md and exit")
    args = parser.parse_args()

    if args.check:
        return run_check()
    if args.rebuild:
        return run_rebuild()
    return run_tui()


if __name__ == "__main__":
    sys.exit(main())
