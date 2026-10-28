"""tingle analyze|plan|render|triggers — development CLI."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .detect import DEFAULT_MODEL, detect
from .plan import plan
from .profile import Profile, category_labels, taxonomy
from .render import render


def _fmt(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:04.1f}"


def main() -> None:
    p = argparse.ArgumentParser(prog="tingle")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("triggers", help="list the trigger taxonomy")
    for name in ("analyze", "plan", "render"):
        s = sub.add_parser(name)
        s.add_argument("video", type=Path)
        s.add_argument("--model", default=DEFAULT_MODEL)
        s.add_argument("--chunk", type=float, default=60.0, help="seconds per Muse Spark request")
        if name != "analyze":
            s.add_argument("--profile", type=Path, required=True, help="JSON {trigger or category: like|dislike}")
            s.add_argument("--no-filter", action="store_true", help="mute instead of filter")
        if name == "render":
            s.add_argument("-o", "--out", type=Path)
    a = p.parse_args()

    if a.cmd == "triggers":
        cats = category_labels()
        for t in taxonomy().values():
            print(f"{t.id:28} {cats[t.category]} / {t.label}")
        return

    events = detect(a.video, a.model, a.chunk)
    if a.cmd == "analyze":
        for e in events:
            print(f"{_fmt(e.start):>7} – {_fmt(e.end):>7}  {e.trigger:26} {e.confidence:.2f}  {e.note}")
        out = a.video.with_suffix(".triggers.json")
        out.write_text(json.dumps([asdict(e) for e in events], indent=1))
        print(f"{len(events)} events → {out}")
        return

    steps = plan(events, Profile.load(a.profile), can_filter=not a.no_filter)
    for s in steps:
        print(f"{_fmt(s.start):>7} – {_fmt(s.end):>7}  {s.action:6} {', '.join(s.triggers):40} {s.reason}")
    if a.cmd == "render":
        out = a.out or a.video.with_name(a.video.stem + ".tuned.mp4")
        render(a.video, steps, out)
        print(f"→ {out}")
