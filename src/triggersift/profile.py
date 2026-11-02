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
    id: str  # the sound class ("tapping_sharp"); it is its own category
    category: str
    sub: str  # "" (v1 had "category.sub" ids; the code still accepts them via id.split(".")[0])
    label: str
    sound: str  # SAM Audio / PE-A-Frame text prompt
    group: str = ""  # UI heading only
    covers: str = ""  # what belongs in the class, shown while labeling


KEYS = {"group", "label", "sound", "covers"}


@cache
def taxonomy() -> dict[str, Trigger]:
    out = {}
    for cid, s in yaml.safe_load(TAXONOMY_FILE.read_text()).items():
        # an unquoted comma inside a flow mapping silently splits a value into extra keys
        if set(s) != KEYS:
            raise ValueError(f"taxonomy.yaml: {cid} has keys {sorted(s)}, expected {sorted(KEYS)}; quote values with commas")
        out[cid] = Trigger(cid, cid, "", s["label"], s["sound"], s["group"], s["covers"])
    return out


@cache
def category_labels() -> dict[str, str]:
    return {t.id: t.label for t in taxonomy().values()}


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
