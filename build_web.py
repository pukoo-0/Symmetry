"""
build_web.py — makes the single-file web app (docs/index.html) from web/index.src.html.

The Python code (symmetry_core.py, web/bridge.py) and the example molecules are embedded in the page,
so docs/index.html works on its own: open it in a browser, host it on GitHub Pages, or serve it from Streamlit.
Run this again after changing any of those files:   python build_web.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = ["symmetry_core.py", os.path.join("web", "bridge.py")] + sorted(
    os.path.join("examples", f) for f in os.listdir(os.path.join(HERE, "examples")) if f.endswith(".xyz"))
MARKER = "/*__PY_FILES__*/{}"


def build() -> str:
    files = {os.path.basename(f): open(os.path.join(HERE, f), encoding="utf-8").read() for f in FILES}
    template = open(os.path.join(HERE, "web", "index.src.html"), encoding="utf-8").read()
    assert MARKER in template, "the placeholder for the embedded files is missing from web/index.src.html"
    return template.replace(MARKER, json.dumps(files, ensure_ascii=False).replace("</", "<\\/"), 1)


def main():
    html = build()
    os.makedirs(os.path.join(HERE, "docs"), exist_ok=True)
    out = os.path.join(HERE, "docs", "index.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"Wrote {out} ({len(html) // 1024} KB)")


if __name__ == "__main__":
    main()
