# TriggerSift

Find every ASMR trigger in a video, then skip or filter out the ones you don't like.

- **Detect**: Muse Wonder listens to the soundtrack (60 s chunks) and labels triggers from `taxonomy.yaml`
  (category → subcategory, e.g. tapping → tapping on leather) with timestamps.
- **Profile**: mark categories or subcategories as like / neutral / dislike.
- **Plan**: long continuous disliked sections are skipped; short or sparse ones (or ones overlapping a liked
  trigger) are filtered out of the audio. Filtering will use SAM Audio; until then it mutes.

## Usage
```
uv run sift triggers                                  # list trigger ids
uv run sift analyze video.mp4                         # → video.triggers.json
uv run sift plan video.mp4 --profile profile.json
uv run sift render video.mp4 --profile profile.json   # → video.tuned.mp4
```
`profile.json`: `{"mouth": "dislike", "mouth.whisper": "like", "crinkling.plastic_bag": "dislike"}`

Needs `ffmpeg` and `MODEL_API_KEY` (env var or `.env`). Results are cached per chunk in `runs/`.
