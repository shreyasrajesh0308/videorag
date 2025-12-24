from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .entity_vectors import answer_with_context, build_entity_vector_index, format_time, retrieve_entities
from .evidence import generate_evidence
from .frames import extract_frames
from .paths import RunPaths, make_run_paths
from .summaries import summarize_captions
from .util import get_openai_api_key, normalize_path, print_stage, read_json, sha256_file, write_json


@dataclass(frozen=True)
class VideoRAGConfig:
    fps: float = 1.0
    window_s: float = 2.0
    stride_s: float = 2.0
    summary_interval_s: float = 10.0
    scale_width: int = 640
    vision_model: str = "gpt-4o-mini"
    summary_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    answer_model: str = "gpt-4o-mini"


class VideoRAG:
    def __init__(
        self,
        video_path: str | Path,
        *,
        fps: float = 1.0,
        runs_root: str | Path = "artifacts/runs",
        env_file: str | Path | None = ".env",
        config: VideoRAGConfig | None = None,
    ) -> None:
        self.video_path = Path(video_path)
        self.runs_root = Path(runs_root)
        self.env_file = Path(env_file) if env_file else None
        self.config = config or VideoRAGConfig(fps=float(fps))
        if fps != self.config.fps:
            self.config = VideoRAGConfig(**{**self.config.__dict__, "fps": float(fps)})

        if not self.video_path.exists():
            raise FileNotFoundError(self.video_path)

        self.video_sha256 = sha256_file(self.video_path)
        run_id = f"{self.video_sha256[:12]}_fps{self.config.fps:g}"
        self.paths: RunPaths = make_run_paths(self.runs_root / run_id)

    def _ensure_manifest(self, overwrite: bool) -> None:
        self.paths.run_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "created_at_unix": int(time.time()),
            "video_path": normalize_path(self.video_path),
            "video_sha256": self.video_sha256,
            "fps": self.config.fps,
            "window_s": self.config.window_s,
            "stride_s": self.config.stride_s,
            "scale_width": self.config.scale_width,
            "vision_model": self.config.vision_model,
            "summary_model": self.config.summary_model,
            "embedding_model": self.config.embedding_model,
            "answer_model": self.config.answer_model,
        }

        if self.paths.manifest_json.exists() and not overwrite:
            existing = read_json(self.paths.manifest_json)
            if (
                existing.get("video_sha256") != manifest["video_sha256"]
                or float(existing.get("fps", -1)) != float(manifest["fps"])
            ):
                raise RuntimeError(
                    f"Run dir manifest mismatch: {self.paths.run_dir} (pass overwrite=True to rebuild)"
                )
            return
        write_json(self.paths.manifest_json, manifest)

    def index(self, *, overwrite: bool = False, preview: bool = True) -> RunPaths:
        self._ensure_manifest(overwrite)

        api_key = get_openai_api_key(self.env_file)
        progress = print

        # 1) Frames
        print_stage("1. FRAME CREATION")
        if overwrite or not self.paths.frames_index_jsonl.exists():
            progress(f"Extracting frames @ {self.config.fps:g} fps → {normalize_path(self.paths.frames_dir)}")
            extract_frames(
                video_path=self.video_path,
                fps=self.config.fps,
                out_dir=self.paths.frames_dir,
                scale_width=self.config.scale_width,
                preview=preview,
            )
            progress(f"Frame index: {normalize_path(self.paths.frames_index_jsonl)}")
        else:
            progress(f"Frames already exist; skipping ({normalize_path(self.paths.frames_dir)})")

        # 2) Evidence (vision)
        print_stage("2. EVIDENCE FORMATION")
        generate_evidence(
            api_key=api_key,
            index_path=self.paths.frames_index_jsonl,
            evidence_path=self.paths.evidence_jsonl,
            state_path=self.paths.state_json,
            errors_path=self.paths.errors_jsonl,
            model=self.config.vision_model,
            window_s=self.config.window_s,
            stride_s=self.config.stride_s,
            summary_interval_s=self.config.summary_interval_s,
            summary_model=self.config.summary_model,
            overwrite=overwrite,
            progress_cb=progress,
        )
        progress(f"Evidence: {normalize_path(self.paths.evidence_jsonl)}")

        # 3) Global caption summary
        print_stage("3. SUMMARISATION")
        summarize_captions(
            api_key=api_key,
            evidence_jsonl=self.paths.evidence_jsonl,
            out_json=self.paths.global_summary_json,
            errors_jsonl=self.paths.errors_jsonl,
            model=self.config.summary_model,
            overwrite=overwrite,
            progress_cb=progress,
        )
        progress(f"Global summary: {normalize_path(self.paths.global_summary_json)}")

        # 4) Entity vector index
        print_stage("4. INDEXING (ENTITY VECTORS)")
        build_entity_vector_index(
            api_key=api_key,
            evidence_jsonl=self.paths.evidence_jsonl,
            state_json=self.paths.state_json,
            out_npz=self.paths.entity_vectors_npz,
            out_meta_json=self.paths.entity_vectors_meta_json,
            embedding_model=self.config.embedding_model,
            overwrite=overwrite,
            progress_cb=progress,
        )
        progress(f"Entity vectors: {normalize_path(self.paths.entity_vectors_npz)}")
        progress(f"Run directory: {normalize_path(self.paths.run_dir)}")

        return self.paths

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        max_mentions: int = 6,
        include_global: bool = True,
    ) -> dict[str, Any]:
        api_key = get_openai_api_key(self.env_file)
        payload = retrieve_entities(
            api_key=api_key,
            vectors_npz=self.paths.entity_vectors_npz,
            meta_json=self.paths.entity_vectors_meta_json,
            query=query,
            top_k=top_k,
            max_mentions=max_mentions,
            embedding_model=self.config.embedding_model,
        )
        global_summary = ""
        if include_global and self.paths.global_summary_json.exists():
            try:
                global_summary = str(read_json(self.paths.global_summary_json).get("summary", "")).strip()
            except Exception:
                global_summary = ""
        payload["global_summary"] = global_summary
        payload["run_dir"] = normalize_path(self.paths.run_dir)
        return payload

    def _local_context_text(self, retrieval: dict[str, Any]) -> str:
        lines: list[str] = []
        lines.append("Query-specific information (local retrieval; timestamped video time):")
        for e in retrieval.get("results", []) or []:
            if not isinstance(e, dict):
                continue
            eid = e.get("entity_id")
            etype = e.get("type", "other")
            desc = e.get("descriptor", "")
            score = e.get("score")
            header = f"- {eid} ({etype})"
            if isinstance(desc, str) and desc.strip():
                header += f": {desc.strip()}"
            if isinstance(score, (int, float)):
                header += f" [score={float(score):.3f}]"
            lines.append(header)
            mentions = e.get("mentions") or []
            if not isinstance(mentions, list):
                continue
            for m in mentions:
                if not isinstance(m, dict):
                    continue
                start_s = float(m.get("start_s", 0.0))
                end_s = float(m.get("end_s", 0.0))
                cap = m.get("caption", "")
                if not isinstance(cap, str):
                    cap = ""
                lines.append(f"  - [{format_time(start_s)}–{format_time(end_s)}] {cap}".strip())
        return "\n".join(lines).strip()

    def answer(
        self,
        query: str,
        *,
        top_k: int = 5,
        max_mentions: int = 6,
        include_global: bool = True,
    ) -> str:
        api_key = get_openai_api_key(self.env_file)
        retrieval = self.retrieve(query, top_k=top_k, max_mentions=max_mentions, include_global=include_global)
        global_summary = retrieval.get("global_summary", "") if isinstance(retrieval.get("global_summary"), str) else ""
        local_context = self._local_context_text(retrieval)
        return answer_with_context(
            api_key=api_key,
            query=query,
            global_summary=global_summary,
            local_context=local_context,
            answer_model=self.config.answer_model,
        )
