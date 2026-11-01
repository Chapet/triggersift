"""Collect varied ASMR videos with chapters from YouTube, for hand labeling and chapter-based training clips.

    uv run python -m triggersift.harvest --target 30

Searches per trigger type, keeps 5–30 min videos with ≥ 3 chapters, at most 2 per channel, downloads at ≤ 720p into
the library (SourceVideos) and writes "<name>.chapters.txt" next to each video. Videos stay local (not in the repo).
"""

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path

LIBRARY = Path("~/Videos/ASMR/SourceVideos").expanduser()
QUERIES = [
    "asmr mouth sounds", "asmr tongue clicking", "asmr whispering ear to ear", "asmr inaudible whispering",
    "asmr mic brushing", "asmr mic scratching", "asmr fluffy mic", "asmr crinkle sounds", "asmr plastic crinkles",
    "asmr tapping glass", "asmr tapping wood", "asmr nail tapping", "asmr hand sounds", "asmr keyboard typing",
    "asmr page turning", "asmr liquid sounds", "asmr water sounds", "asmr slime", "asmr soap cutting",
    "asmr kinetic sand", "asmr eating", "asmr rain sounds tapping", "asmr fire crackling", "asmr hair brushing",
    "asmr scissors haircut", "asmr lid sounds", "asmr foam squeezing", "asmr breathing", "asmr trigger assortment",
    "asmr no talking triggers", "asmr soft spoken", "asmr layered sounds",
]
MIN_S, MAX_S, MIN_CHAPTERS, PER_CHANNEL = 300, 1800, 3, 2


def run(*args: str, timeout: int = 180) -> str:
    return subprocess.run(["yt-dlp", *args], capture_output=True, text=True, timeout=timeout).stdout


def candidates(query: str, n: int = 25) -> list[tuple[str, int]]:
    out = run("--flat-playlist", "--print", "%(id)s %(duration)s", f"ytsearch{n}:{query}")
    ids = []
    for line in out.splitlines():
        vid, _, dur = line.partition(" ")
        if dur.replace(".", "").isdigit() and MIN_S <= float(dur) <= MAX_S:
            ids.append((vid, int(float(dur))))
    return ids


def chapters_txt(info: dict) -> str:
    def fmt(t):
        t = int(t)
        return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60}:{t % 60:02d}"
    lines = [f"# Chapters: {info['title']} (https://www.youtube.com/watch?v={info['id']})",
             f"# Channel: {info.get('channel')}", ""]
    lines += [f"{fmt(c['start_time'])} {c['title']}" for c in info["chapters"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=30, help="number of videos to download")
    a = ap.parse_args()
    LIBRARY.mkdir(parents=True, exist_ok=True)
    have = {p.stem.rsplit("[", 1)[-1].rstrip("]") for p in LIBRARY.iterdir()}
    per_channel: Counter = Counter()
    seen: set[str] = set()
    got = 0
    # round-robin over the queries so the set stays varied even if we stop early
    pools = {q: candidates(q) for q in QUERIES}
    print(f"{sum(len(v) for v in pools.values())} candidates (5–30 min) from {len(QUERIES)} searches", flush=True)
    while got < a.target and any(pools.values()):
        for q in QUERIES:
            if got >= a.target or not pools[q]:
                continue
            vid, dur = pools[q].pop(0)
            if vid in seen or vid in have:
                continue
            seen.add(vid)
            try:
                info = json.loads(run("-J", "--skip-download", f"https://www.youtube.com/watch?v={vid}"))
            except (json.JSONDecodeError, subprocess.TimeoutExpired):
                continue
            if not isinstance(info, dict):  # unavailable / age-restricted: yt-dlp prints null
                continue
            ch = info.get("channel_id") or info.get("channel")
            if len(info.get("chapters") or []) < MIN_CHAPTERS or per_channel[ch] >= PER_CHANNEL:
                continue
            res = subprocess.run(
                ["yt-dlp", "-q", "--no-warnings", "-f", "bv*[height<=720]+ba/b[height<=720]/b",
                 "-o", str(LIBRARY / "%(title).80s [%(id)s].%(ext)s"), f"https://www.youtube.com/watch?v={vid}"],
                capture_output=True, text=True, timeout=1800)
            files = [p for p in LIBRARY.glob(f"*[[]{vid}[]].*") if p.suffix in {".webm", ".mp4", ".mkv"}]
            if res.returncode != 0 or not files:
                print(f"  failed {vid}: {res.stderr[-200:]}", flush=True)
                continue
            files[0].with_name(files[0].stem + ".chapters.txt").write_text(chapters_txt(info))
            per_channel[ch] += 1
            got += 1
            print(f"[{got}/{a.target}] {dur // 60} min, {len(info['chapters'])} chapters, {info.get('channel')}: "
                  f"{info['title'][:70]}  (query: {q})", flush=True)
    print(f"done: {got} videos from {len(per_channel)} channels", flush=True)


if __name__ == "__main__":
    main()
