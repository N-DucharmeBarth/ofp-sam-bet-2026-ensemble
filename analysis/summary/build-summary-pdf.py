"""Render the HTML report to PDF via headless Chromium.

Print-to-PDF rather than re-authoring in reportlab: the HTML already carries the
type scale, tables and figures, and the `@media print` block in report.html
supplies paper-specific paging (A4, light palette, break-inside rules). Building
the same document a second time in a different engine would only let the two
drift apart.

Requires playwright + the preinstalled Chromium at /opt/pw-browsers/chromium.
"""

import os
import sys

OUT = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")),
    "out")

SRC = os.path.join(OUT, "bet-2026-tag-reporting-rate-summary.html")
DST = os.path.join(OUT, "bet-2026-tag-reporting-rate-summary.pdf")
CHROMIUM = "/opt/pw-browsers/chromium"

FOOT = """
<div style="width:100%;font-family:system-ui,sans-serif;font-size:7.5pt;
            color:#6c7980;padding:0 15mm;display:flex;
            justify-content:space-between;">
  <span>BET 2026 ensemble &middot; tag reporting-rate findings</span>
  <span>Page <span class="pageNumber"></span> of <span class="totalPages"></span></span>
</div>
"""


def main():
    from playwright.sync_api import sync_playwright

    if not os.path.exists(SRC):
        raise SystemExit(f"missing {SRC} -- run analysis/summary/build-summary.py first")

    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=CHROMIUM if os.path.exists(CHROMIUM) else None
        )
        # force the light theme so the PDF never inherits a dark host setting
        page = browser.new_page(color_scheme="light")
        page.goto("file://" + SRC)
        page.emulate_media(media="print", color_scheme="light")
        page.wait_for_load_state("networkidle")
        page.pdf(
            path=DST,
            format="A4",
            print_background=True,
            prefer_css_page_size=True,
            display_header_footer=True,
            header_template="<div></div>",
            footer_template=FOOT,
        )
        browser.close()

    print(f"wrote {os.path.relpath(DST, os.path.dirname(OUT))} "
          f"({os.path.getsize(DST)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
