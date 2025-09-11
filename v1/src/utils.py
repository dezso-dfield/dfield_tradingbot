import json, time
from pathlib import Path
from rich import print

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def save_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def sleep_polite(sec=1.0):
    time.sleep(sec)