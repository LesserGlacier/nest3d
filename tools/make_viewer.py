"""Inline a viewer payload into the viewer template."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _settle_track(payload, traced):
    """Re-index a traced settle onto the host payload's part order.

    Both come from the same STEP files, but a trace is its own run and
    there is nothing that guarantees ``load_parts`` handed them out in the
    same order.  Names are the thing both agree on, so map through those
    and fail loudly if the sets differ at all -- a silently mismatched
    frame would draw the right box with the wrong parts in it.
    """
    want = [p["name"] for p in payload["parts"]]
    have = [p["name"] for p in traced["parts"]]
    if sorted(want) != sorted(have):
        raise SystemExit("--settle-from has different parts; refusing to merge")
    where = {n: i for i, n in enumerate(have)}
    pick = [where[n] for n in want]
    out = []
    for fr in traced["settle"]["frames"]:
        f = dict(fr)
        f["m"] = [fr["m"][i] for i in pick]
        f["order"] = list(range(len(want)))
        out.append(f)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="?", default=str(ROOT / "viewer_data.json"))
    ap.add_argument("--template", default=str(Path(__file__).parent / "viewer_template.html"))
    ap.add_argument("--title")
    ap.add_argument("--search-from",
                    help="second payload whose search track (one recorded "
                         "chain's actual walk) is merged into this one")
    ap.add_argument("--settle-from",
                    help="payload from trace_settle.py, whose pass-by-pass "
                         "settle is merged in as a fourth track")
    ap.add_argument("--out", default=str(ROOT / "viewer.html"))
    args = ap.parse_args()

    payload = json.loads(Path(args.data).read_text())
    if args.search_from:
        other = json.loads(Path(args.search_from).read_text())
        if [p["name"] for p in other["parts"]] != [p["name"] for p in payload["parts"]]:
            raise SystemExit("--search-from has different parts; refusing to merge")
        payload["tracks"]["search"] = other["tracks"]["search"]
        payload["stats"]["walk_evaluated"] = other["stats"]["evaluated"]
    if args.settle_from:
        other = json.loads(Path(args.settle_from).read_text())
        payload["tracks"]["settle"] = _settle_track(payload, other)
    html = Path(args.template).read_text(encoding="utf-8")

    # </script> inside a JSON string would close the host script tag early.
    blob = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    html = html.replace("__DATA__", blob)
    if args.title:
        html = html.replace("<title>Nesting Replay</title>",
                            "<title>%s</title>" % args.title)

    Path(args.out).write_text(html, encoding="utf-8")
    mb = Path(args.out).stat().st_size / 1e6
    print("wrote %s  (%.1f MB)" % (args.out, mb))
    print("  parts   %d" % len(payload["parts"]))
    if payload["tracks"].get("settle"):
        print("  settle  %d frames" % len(payload["tracks"]["settle"]))
    print("  best    %d frames" % len(payload["tracks"]["best"]))
    print("  search  %d frames" % len(payload["tracks"]["search"]))
    print("  from    %d evaluations" % payload["stats"]["evaluated"])


if __name__ == "__main__":
    main()
