from __future__ import annotations

import subprocess
from pathlib import Path

from .util import write_frame_index


def extract_frames(
    *,
    video_path: Path,
    fps: float,
    out_dir: Path,
    scale_width: int = 640,
    jpeg_quality: int = 3,
    preview: bool = True,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "frame_%05d.jpg")
    vf = f"fps={fps},scale={scale_width}:-1"

    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            vf,
            "-q:v",
            str(jpeg_quality),
            pattern,
        ],
        check=True,
    )

    frame_paths = sorted(out_dir.glob("frame_*.jpg"))
    index_jsonl, _ = write_frame_index(out_dir=out_dir, fps=fps, frame_paths=frame_paths)

    if preview:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(video_path),
                "-vf",
                f"fps={fps},scale=200:-1,tile=10x10",
                "-frames:v",
                "1",
                str(out_dir / "preview.jpg"),
            ],
            check=True,
        )

    return index_jsonl
