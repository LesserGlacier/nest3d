"""Inline a viewer payload into the viewer template."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="?", default=str(ROOT / "viewer_data.json"))
    ap.add_argument("--template", default=str(Path(__file__).parent / "viewer_template.html"))
    ap.add_argument("--title")
    ap.add_argument("--search-from",
                    help="second payload whose search track (one recorded "
                         "chain's actual walk) is merged into this one")
    ap.add_argument("--out", default=str(ROOT / "viewer.html"))
    args = ap.parse_args()

    payload = json.loads(Path(args.data).read_text())
    if args.search_from:
        other = json.loads(Path(args.search_from).read_text())
        if [p["name"] for p in other["parts"]] != [p["name"] for p in payload["parts"]]:
            raise SystemExit("--search-from has different parts; refusing to merge")
        payload["tracks"]["search"] = other["tracks"]["search"]
        payload["stats"]["walk_evaluated"] = other["stats"]["evaluated"]
    html = Path(args.template).read_text(encoding="utf-8")

    # </script> inside a JSON string would close the host script tag early.
    blob = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    html = html.replace("__DATA__", blob)
    if args.title:
        html = html.replace("<title>Glass Positives Nesting</title>",
                            "<title>%s</title>" % args.title)

    Path(args.out).write_text(html, encoding="utf-8")
    mb = Path(args.out).stat().st_size / 1e6
    print("wrote %s  (%.1f MB)" % (args.out, mb))
    print("  parts   %d" % len(payload["parts"]))
    print("  best    %d frames" % len(payload["tracks"]["best"]))
    print("  search  %d frames" % len(payload["tracks"]["search"]))
    print("  from    %d evaluations" % payload["stats"]["evaluated"])


if __name__ == "__main__":
    main()
