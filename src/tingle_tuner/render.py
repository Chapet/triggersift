"""Render a cleaned copy of the video with ffmpeg: cut skipped sections, silence muted ones.

FILTER steps are rendered as mutes until SAM Audio separation is wired in.
"""

import subprocess
from pathlib import Path

from .plan import Action, Step

FADE_S = 0.05


def _mute_expr(spans: list[tuple[float, float]]) -> str:
    """ffmpeg volume expression: 0 inside spans, 1 outside, linear FADE_S ramps at the edges."""
    # gain for one span = clip(distance to the span / FADE_S, 0, 1); spans multiply
    parts = [f"clip(max({s}-t,t-{e})/{FADE_S},0,1)" for s, e in spans]
    return "*".join(parts) if parts else "1"


def render(src: Path, steps: list[Step], dst: Path) -> None:
    skips = [(s.start, s.end) for s in steps if s.action == Action.SKIP]
    mutes = [(s.start, s.end) for s in steps if s.action in (Action.MUTE, Action.FILTER)]
    keep = "*".join(f"not(between(t,{s},{e}))" for s, e in skips) or "1"
    vf = f"select='{keep}',setpts=N/FRAME_RATE/TB"
    # mute before cutting so the timestamps in the expression are the original ones
    af = f"volume='{_mute_expr(mutes)}':eval=frame,aselect='{keep}',asetpts=N/SR/TB"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", vf, "-af", af,
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "aac", "-b:a", "192k", str(dst)],
        check=True)
