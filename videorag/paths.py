from __future__ import annotations

import dataclasses
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    manifest_json: Path
    frames_dir: Path
    frames_index_jsonl: Path
    frames_preview_jpg: Path
    evidence_dir: Path
    evidence_jsonl: Path
    state_json: Path
    errors_jsonl: Path
    summaries_dir: Path
    global_summary_json: Path
    vector_dir: Path
    entity_vectors_npz: Path
    entity_vectors_meta_json: Path


def make_run_paths(run_dir: Path) -> RunPaths:
    frames_dir = run_dir / "frames"
    evidence_dir = run_dir / "evidence"
    summaries_dir = run_dir / "summaries"
    vector_dir = run_dir / "vector"
    return RunPaths(
        run_dir=run_dir,
        manifest_json=run_dir / "manifest.json",
        frames_dir=frames_dir,
        frames_index_jsonl=frames_dir / "index.jsonl",
        frames_preview_jpg=frames_dir / "preview.jpg",
        evidence_dir=evidence_dir,
        evidence_jsonl=evidence_dir / "evidence.jsonl",
        state_json=evidence_dir / "state.json",
        errors_jsonl=evidence_dir / "errors.jsonl",
        summaries_dir=summaries_dir,
        global_summary_json=summaries_dir / "global_summary.json",
        vector_dir=vector_dir,
        entity_vectors_npz=vector_dir / "entities.npz",
        entity_vectors_meta_json=vector_dir / "entities.json",
    )
