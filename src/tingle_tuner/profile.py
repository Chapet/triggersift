"""Trigger taxonomy and the viewer's like/dislike profile."""

import json
from dataclasses import dataclass, field
from enum import StrEnum
from functools import cache
from pathlib import Path

import yaml

TAXONOMY_FILE = Path(__file__).with_name("taxonomy.yaml")


class Pref(StrEnum):
    LIKE = "like"
    NEUTRAL = "neutral"
    DISLIKE = "dislike"


@dataclass(frozen=True)
class Trigger:
    id: str  # "category.sub"
    category: str
    sub: str
    label: str
    sound: str  # SAM Audio text prompt


@cache
def taxonomy() -> dict[str, Trigger]:
    data = yaml.safe_load(TAXONOMY_FILE.read_text())
    out = {}
    for cat, spec in data.items():
        for sub, s in spec["subs"].items():
            tid = f"{cat}.{sub}"
            out[tid] = Trigger(tid, cat, sub, s["label"], s["sound"])
    return out


@cache
def category_labels() -> dict[str, str]:
    return {cat: spec["label"] for cat, spec in yaml.safe_load(TAXONOMY_FILE.read_text()).items()}


@dataclass
class Profile:
    """Preferences by category ("mouth") or subcategory ("mouth.kissing"); subcategory wins."""

    prefs: dict[str, Pref] = field(default_factory=dict)

    def pref(self, trigger_id: str) -> Pref:
        if trigger_id in self.prefs:
            return self.prefs[trigger_id]
        return self.prefs.get(trigger_id.split(".")[0], Pref.NEUTRAL)

    @classmethod
    def load(cls, path: Path) -> "Profile":
        return cls({k: Pref(v) for k, v in json.loads(path.read_text()).items()})

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({k: str(v) for k, v in self.prefs.items()}, indent=2) + "\n")
