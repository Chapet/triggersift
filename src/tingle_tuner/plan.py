"""Turn detected events + a profile into actions: skip, filter or mute each disliked stretch."""

from dataclasses import dataclass
from enum import StrEnum

from .detect import Event
from .profile import Pref, Profile


class Action(StrEnum):
    SKIP = "skip"  # jump over the section
    FILTER = "filter"  # remove only that sound (SAM Audio)
    MUTE = "mute"  # silence the section (fallback when filtering isn't available)


@dataclass
class Step:
    start: float
    end: float
    action: Action
    triggers: list[str]
    reason: str


MERGE_GAP_S = 2.0
SKIP_MIN_S = 15.0
SKIP_MIN_COVERAGE = 0.5


def _runs(events: list[Event]) -> list[tuple[float, float, list[Event]]]:
    """Group events whose gaps are under MERGE_GAP_S."""
    runs: list[tuple[float, float, list[Event]]] = []
    for e in sorted(events, key=lambda e: e.start):
        if runs and e.start - runs[-1][1] < MERGE_GAP_S:
            s, t, evs = runs[-1]
            runs[-1] = (s, max(t, e.end), evs + [e])
        else:
            runs.append((e.start, e.end, [e]))
    return runs


def _covered(events: list[Event]) -> float:
    """Seconds covered by the union of the events."""
    total, cur_s, cur_t = 0.0, None, None
    for e in sorted(events, key=lambda e: e.start):
        if cur_t is None or e.start > cur_t:
            if cur_t is not None:
                total += cur_t - cur_s
            cur_s, cur_t = e.start, e.end
        else:
            cur_t = max(cur_t, e.end)
    return total + ((cur_t - cur_s) if cur_t is not None else 0.0)


def plan(events: list[Event], profile: Profile, can_filter: bool = True,
         min_confidence: float = 0.4) -> list[Step]:
    events = [e for e in events if e.confidence >= min_confidence]
    disliked = [e for e in events if profile.pref(e.trigger) == Pref.DISLIKE]
    liked = [e for e in events if profile.pref(e.trigger) == Pref.LIKE]
    soft = Action.FILTER if can_filter else Action.MUTE
    steps = []
    for start, end, evs in _runs(disliked):
        triggers = sorted({e.trigger for e in evs})
        length = end - start
        coverage = _covered(evs) / length if length else 0.0
        overlaps_liked = any(l.start < end and l.end > start for l in liked)
        if length >= SKIP_MIN_S and coverage >= SKIP_MIN_COVERAGE and not overlaps_liked:
            steps.append(Step(start, end, Action.SKIP, triggers,
                              f"{length:.0f}s of disliked sound ({coverage:.0%} covered)"))
        else:
            why = ("liked trigger plays at the same time" if overlaps_liked
                   else f"short or sparse ({length:.0f}s, {coverage:.0%} covered)")
            steps.append(Step(start, end, soft, triggers, why))
    return steps
