from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable


def clamp_text(s: str, limit: int) -> str:
    s = (s or "").strip()
    if len(s) <= limit:
        return s
    return s[: limit - 3] + "..."


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_dotenv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    out: dict[str, str] = {}
    for m in re.finditer(r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*("?)(.*?)\2\s*$', text, flags=re.MULTILINE):
        out[m.group(1)] = m.group(3)
    return out


def get_openai_api_key(env_file: Path | None = None) -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    if env_file:
        key = load_dotenv(env_file).get("OPENAI_API_KEY", "").strip()
        if key:
            return key
    raise RuntimeError("OPENAI_API_KEY not set (export it or put it in .env).")


def write_frame_index(
    *,
    out_dir: Path,
    fps: float,
    frame_paths: list[Path],
) -> tuple[Path, Path]:
    def fmt_ts(t: float) -> str:
        m = int(t // 60)
        s = t - 60 * m
        return f"{m:02d}:{s:06.3f}"

    index_jsonl = out_dir / "index.jsonl"
    index_csv = out_dir / "index.csv"

    with index_jsonl.open("w", encoding="utf-8") as f:
        for i, path in enumerate(frame_paths, start=1):
            t = (i - 1) / fps
            rec = {
                "frame_index": i,
                "timestamp_s": round(t, 3),
                "timestamp": fmt_ts(t),
                "path": str(path.resolve()),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with index_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_index", "timestamp_s", "timestamp", "path"])
        for i, path in enumerate(frame_paths, start=1):
            t = (i - 1) / fps
            w.writerow([i, round(t, 3), fmt_ts(t), str(path)])

    return index_jsonl, index_csv


def normalize_path(path: Path) -> str:
    # Prefer workspace-relative paths if possible, else absolute.
    try:
        return str(path.relative_to(Path.cwd()))
    except Exception:
        return str(path)


def batched(xs: list[Any], n: int) -> Iterable[list[Any]]:
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def print_stage(title: str) -> None:
    bar = "=" * len(title)
    print(f"\n{title}\n{bar}", flush=True)


def format_progress(i: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return f"{i}"
    i = max(0, min(i, total))
    filled = int(round(width * (i / total)))
    return f"[{'#' * filled}{'.' * (width - filled)}] {i}/{total} ({(100 * i / total):.1f}%)"
