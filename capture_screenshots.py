"""Capture dashboard screenshots for Bab 7 of the Final Project report.

Needs a running dashboard (default http://localhost:8503) and Playwright with the installed Chrome:
    .venv-dashboard/Scripts/python.exe SourceCode/dashboard/capture_screenshots.py [base_url]
Writes PNGs to docs/final project/figures/.
"""
from pathlib import Path
import sys

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs/final project/figures'
BASE = sys.argv[1] if len(sys.argv) > 1 else 'http://localhost:8503'

# (file name, page parameter, element text the capture starts at, element text it ends before, or None for the end)
SHOTS = [
    ('dashboard_ringkasan.png', 'ringkasan', 'Permintaan Yellow Taxi', None),
    ('dashboard_prediksi_atas.png', 'prediksi', 'Prediksi pickup satu jam', 'Error menurut tingkat'),
    ('dashboard_prediksi_bawah.png', 'prediksi', 'Error menurut tingkat', None),
    ('dashboard_jaringan_peta.png', 'jaringan', 'Jaringan perjalanan', '10 koridor terbesar'),
    ('dashboard_jaringan_tabel.png', 'jaringan', '10 koridor terbesar', None),
    ('dashboard_insight_ml.png', 'insight', '1 · Prediksi cukup andal', '3 · Bandara'),
    ('dashboard_insight_graph.png', 'insight', '3 · Bandara', None),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel='chrome', args=['--use-angle=swiftshader', '--enable-unsafe-swiftshader'])
        page = browser.new_page(viewport={'width': 1500, 'height': 900}, device_scale_factor=2)
        for name, key, start, end in SHOTS:
            page.goto(f'{BASE}/?page={key}')
            page.get_by_text(start, exact=False).first.wait_for(timeout=60000)
            page.wait_for_timeout(9000)  # map tiles and WebGL layers need real time to draw
            # Make the whole app one tall page so a single screenshot covers the section.
            height = page.evaluate("document.querySelector('[data-testid=\"stMain\"]').scrollHeight")
            page.set_viewport_size({'width': 1500, 'height': height + 200})
            page.wait_for_timeout(8000)  # let the taller layout settle before measuring
            top = page.get_by_text(start, exact=False).first.bounding_box()['y'] - 2
            bottom = (page.get_by_text(end, exact=False).first.bounding_box()['y'] - 6) if end else height
            left = page.evaluate("document.querySelector('[data-testid=\"stMain\"]').getBoundingClientRect().left")
            page.screenshot(path=str(OUT / name), clip={'x': left, 'y': top, 'width': 1500 - left,
                                                        'height': bottom - top})
            page.set_viewport_size({'width': 1500, 'height': 900})
            print('saved', name)
        browser.close()


if __name__ == '__main__':
    main()
