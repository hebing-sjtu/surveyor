"""Render a Markdown reply as a PNG.

Neither Feishu nor WeCom draws Markdown properly. Feishu cards understand a
small subset and no tables at all; WeCom's markdown message type drops tables
and formulas outright. A paper note is mostly headings, tables and formulas, so
what arrives is a wall of pipes and dollar signs. An image arrives looking the
way it was written.

The renderer is the browser that is already on the machine, driven headless. It
costs no extra Python dependency, and it draws the MathML that
:mod:`surveyor.render` produces, so formulas survive into chat.

Chrome only photographs the viewport, so the page is loaded twice: once to ask
the layout how tall it turned out, then again with a window that size.
"""

from __future__ import annotations

import html
import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from ..render import to_html

log = logging.getLogger(__name__)

WIDTH = 860
# Chrome refuses absurd window sizes, and a reply this long should be split.
MAX_HEIGHT = 12000
TIMEOUT = 60

_CANDIDATES = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
)
_ON_PATH = (
    "google-chrome", "google-chrome-stable", "chromium", "chromium-browser",
    "microsoft-edge", "brave-browser", "chrome",
)

_HEIGHT_IN_TITLE = re.compile(r"<title>surveyor:(\d+)</title>")


class RenderError(RuntimeError):
    pass


def find_browser() -> str | None:
    """Locate a Chrome-family browser, or None if the machine has none."""
    override = os.environ.get("SURVEYOR_BROWSER")
    if override:
        return override if Path(override).exists() or shutil.which(override) else None
    for name in _ON_PATH:
        if found := shutil.which(name):
            return found
    for path in _CANDIDATES:
        if Path(path).exists():
            return path
    return None


def available() -> bool:
    return find_browser() is not None


# What chat clients get wrong: table rows, formulas, and headings, which they
# print as literal pipes, dollars and hashes.
_NEEDS_PICTURE = re.compile(
    r"^\s*\|.*\|\s*$"          # a table row
    r"|^\s*#{1,6}\s"           # a heading
    r"|\$\$"                   # display math
    r"|\$[^$\n]{2,}\$",        # inline math
    re.MULTILINE,
)
# Below this a reply is a sentence or two, and a picture of it is just slower.
_PICTURE_WORTH_IT = 700


# One picture holds far more than a chat bubble, but a group-robot webhook caps
# the upload at 2 MB, and a very tall image is unpleasant to read on a phone.
PAGE_CHARS = 6000


def pages(text: str) -> list[str]:
    """Split a long reply into the pieces that each become one picture."""
    from .router import split_message

    return split_message(text, PAGE_CHARS)


def should_draw(text: str, mode: str = "auto") -> bool:
    """Whether this particular reply is better sent as a picture."""
    if mode == "never" or not text.strip():
        return False
    if not available():
        return False
    if mode == "always":
        return True
    return bool(_NEEDS_PICTURE.search(text)) or len(text) > _PICTURE_WORTH_IT


# Deliberately not the app's stylesheet: this is a small image read on a phone,
# so it wants a plain light background and larger type than a desktop page.
_CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body {
  margin: 0;
  background: #ffffff;
  color: #16181d;
  font: 15.5px/1.72 -apple-system, "Helvetica Neue", "PingFang SC",
        "Microsoft YaHei", "Noto Sans CJK SC", sans-serif;
}
#page { width: %(width)dpx; padding: 26px 30px 30px; }
h1 { margin: 0 0 14px; font-size: 21px; }
h2 { margin: 22px 0 8px; font-size: 17px; }
h3 { margin: 18px 0 6px; font-size: 15.5px; }
h1, h2, h3 { line-height: 1.35; }
p, li { margin: 8px 0; }
ul, ol { margin: 8px 0; padding-left: 22px; }
blockquote {
  margin: 12px 0;
  padding: 2px 0 2px 14px;
  border-left: 3px solid #d8dbe2;
  color: #565d6b;
}
code {
  padding: 1px 5px;
  border-radius: 5px;
  background: #f1f2f5;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 13px;
}
pre { padding: 12px; border-radius: 8px; background: #f1f2f5; overflow: hidden; }
pre code { padding: 0; background: none; }
a { color: #1f6feb; text-decoration: none; }
hr { border: 0; border-top: 1px solid #e4e6ec; margin: 20px 0; }

/* Tables cannot scroll in an image, so they wrap and shrink instead. */
.table-scroll { margin: 12px 0; overflow: hidden; }
table { width: 100%%; border-collapse: collapse; font-size: 12.5px; table-layout: fixed; }
th, td {
  padding: 6px 8px;
  border: 1px solid #e4e6ec;
  text-align: left;
  vertical-align: top;
  word-break: break-word;
}
th { background: #f7f8fa; font-weight: 600; }

math { font-size: 1.05em; }
.math-block { display: block; margin: 12px 0; text-align: center; }
.math-block math { font-size: 1.1em; }
code.math-raw { color: #565d6b; }

.footer { margin-top: 22px; color: #8a909c; font-size: 12px; }
"""

_PAGE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>surveyor:0</title>
<style>%(css)s</style></head>
<body><div id="page">%(body)s%(footer)s</div>
<script>
document.title = "surveyor:" +
  Math.ceil(document.getElementById("page").getBoundingClientRect().height);
</script>
</body></html>
"""


def build_page(markdown: str, footer: str = "") -> str:
    body = to_html(markdown)
    note = f'<div class="footer">{html.escape(footer)}</div>' if footer else ""
    return _PAGE % {"css": _CSS % {"width": WIDTH}, "body": body, "footer": note}


def _run(browser: str, arguments: list[str], url: str) -> str:
    # Deliberately no --user-data-dir: asking Chrome to build a fresh profile
    # hangs indefinitely on macOS. Headless mode does not disturb the running
    # browser, so sharing the default profile is both safe and fast.
    command = [
        browser,
        "--headless=new",
        "--disable-gpu",
        "--hide-scrollbars",
        "--force-color-profile=srgb",
        *arguments,
        url,
    ]
    try:
        finished = subprocess.run(
            command, capture_output=True, timeout=TIMEOUT, check=False
        )
    except FileNotFoundError as exc:
        raise RenderError(f"{browser} is not executable") from exc
    except subprocess.TimeoutExpired as exc:
        raise RenderError("the browser did not finish in time") from exc
    return finished.stdout.decode("utf-8", errors="replace")


def _measure(browser: str, url: str) -> int:
    """Ask the layout how tall the page came out."""
    dom = _run(browser, ["--dump-dom", "--virtual-time-budget=4000"], url)
    match = _HEIGHT_IN_TITLE.search(dom)
    if not match:
        raise RenderError("could not measure the page height")
    return int(match.group(1))


def markdown_to_png(markdown: str, footer: str = "") -> bytes:
    """Draw a Markdown reply and return the PNG bytes."""
    browser = find_browser()
    if browser is None:
        raise RenderError(
            "no Chrome-family browser found; set SURVEYOR_BROWSER to one"
        )

    with tempfile.TemporaryDirectory(prefix="surveyor-render-") as workspace:
        directory = Path(workspace)
        page = directory / "reply.html"
        page.write_text(build_page(markdown, footer), encoding="utf-8")
        url = page.as_uri()

        height = min(max(_measure(browser, url), 80), MAX_HEIGHT)
        shot = directory / "reply.png"
        _run(
            browser,
            [
                f"--window-size={WIDTH},{height}",
                "--default-background-color=FFFFFFFF",
                "--virtual-time-budget=4000",
                f"--screenshot={shot}",
            ],
            url,
        )
        if not shot.is_file():
            raise RenderError("the browser produced no image")
        return shot.read_bytes()
