"""Web player: pick a video, it gets analyzed ahead of the playhead, disliked triggers are skipped or muted live.

    uv run python -m triggersift.server          # http://localhost:8000
Library folder: $TRIGGERSIFT_LIBRARY (default ~/Videos/ASMR/SourceVideos). Profile: data/profile.json.
"""

import hashlib
import os
import threading
import traceback
from dataclasses import asdict
from pathlib import Path

import uvicorn
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse

from .detect import Chunk, analyze_chunk, cached_chunk, chunks, duration, run_dir
from .labeling import make_routes
from .plan import plan
from .profile import Pref, Profile, category_labels, taxonomy

LIBRARY = Path(os.environ.get("TRIGGERSIFT_LIBRARY", "~/Videos/ASMR/SourceVideos")).expanduser()
DATA = Path("data")
PROFILE = DATA / "profile.json"
VIDEO_EXT = {".mp4", ".webm", ".mkv", ".mov", ".m4v"}
WORKERS = 6

app = FastAPI()


def _videos() -> dict[str, Path]:
    return {hashlib.sha1(p.name.encode()).hexdigest()[:12]: p
            for p in sorted(LIBRARY.iterdir()) if p.suffix.lower() in VIDEO_EXT}


def _video(vid: str) -> Path:
    if (p := _videos().get(vid)) is None:
        raise HTTPException(404, "unknown video")
    return p


def _profile() -> Profile:
    return Profile.load(PROFILE) if PROFILE.exists() else Profile()


class Analysis:
    """Analyzes the chunks of one video with a few workers, always picking the next missing chunk at or after
    the playhead (then wrapping around), so results stay ahead of what is being watched."""

    def __init__(self, path: Path):
        self.path = path
        self.duration = duration(path)
        self.chunks = chunks(self.duration)
        self.run = run_dir(path)
        self.events: dict[int, list] = {}
        self.errors: dict[int, str] = {}
        self.busy: set[int] = set()
        self.focus = 0.0
        self.lock = threading.Lock()
        self.running = 0
        for c in self.chunks:
            if (evs := cached_chunk(self.run, c)) is not None:
                self.events[c.index] = evs

    def start(self, focus: float) -> None:
        with self.lock:
            self.focus = focus
            want = min(WORKERS, len(self._missing())) - self.running
            self.running += max(0, want)
        for _ in range(max(0, want)):
            threading.Thread(target=self._work, daemon=True).start()

    def _missing(self) -> list[Chunk]:
        return [c for c in self.chunks
                if c.index not in self.events and c.index not in self.busy and c.index not in self.errors]

    def _next(self) -> Chunk | None:
        with self.lock:
            missing = self._missing()
            if not missing:
                self.running -= 1
                return None
            # the chunk under the playhead first, then the ones after it, then the ones before it
            c = min(missing, key=lambda c: (1, c.start) if c.start + c.length <= self.focus
                    else (0, max(0.0, c.start - self.focus)))
            self.busy.add(c.index)
            return c

    def _work(self) -> None:
        while (c := self._next()) is not None:
            try:
                evs = analyze_chunk(self.run, self.path, c)
                with self.lock:
                    self.events[c.index] = evs
            except Exception as e:  # keep the other chunks going; the UI shows failed chunks
                traceback.print_exc()
                with self.lock:
                    self.errors[c.index] = str(e)[:300]
            finally:
                with self.lock:
                    self.busy.discard(c.index)

    def state(self) -> dict:
        with self.lock:
            events = sorted((e for evs in self.events.values() for e in evs), key=lambda e: e.start)
            done = sorted(self.events)
            busy = sorted(self.busy)
            errors = dict(self.errors)
        return {
            "duration": self.duration,
            "chunks": [{"index": c.index, "start": c.start, "end": c.start + c.length,
                        "status": "done" if c.index in done else "busy" if c.index in busy
                        else "error" if c.index in errors else "todo"} for c in self.chunks],
            "events": [asdict(e) for e in events],
            # the player mutes "filter" steps until SAM Audio is in
            "plan": [asdict(s) for s in plan(events, _profile())],
        }


_analyses: dict[str, Analysis] = {}
_analyses_lock = threading.Lock()


def _analysis(vid: str) -> Analysis:
    with _analyses_lock:
        if vid not in _analyses:
            _analyses[vid] = Analysis(_video(vid))
        return _analyses[vid]


app.include_router(make_routes(_video))


@app.get("/")
def index():
    return FileResponse(Path(__file__).with_name("web") / "index.html")


@app.get("/api/videos")
def videos():
    out = []
    for vid, p in _videos().items():
        a = _analyses.get(vid)
        out.append({"id": vid, "name": p.stem, "analyzed": len(a.events) if a else None,
                    "chunks": len(a.chunks) if a else None})
    return out


@app.get("/api/videos/{vid}/media")
def media(vid: str):
    return FileResponse(_video(vid))  # Starlette serves Range requests, so seeking works


@app.post("/api/videos/{vid}/analyze")
def analyze(vid: str, focus: float = 0.0):
    a = _analysis(vid)
    a.start(focus)
    return {"ok": True}


@app.get("/api/videos/{vid}/state")
def state(vid: str):
    return _analysis(vid).state()


@app.get("/api/taxonomy")
def get_taxonomy():
    cats = category_labels()
    out = {c: {"label": label, "subs": {}} for c, label in cats.items()}
    for t in taxonomy().values():
        out[t.category]["subs"][t.id] = t.label
    return out


@app.get("/api/profile")
def get_profile():
    return {k: str(v) for k, v in _profile().prefs.items()}


@app.put("/api/profile")
def put_profile(prefs: dict[str, str] = Body(...)):
    known = set(taxonomy()) | set(category_labels())
    # "neutral" is kept: on a subcategory it overrides a disliked/liked category
    clean = {k: Pref(v) for k, v in prefs.items() if k in known and v in {p.value for p in Pref}}
    DATA.mkdir(exist_ok=True)
    Profile(clean).save(PROFILE)
    return {k: str(v) for k, v in clean.items()}


def main() -> None:
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8000")))


if __name__ == "__main__":
    main()
