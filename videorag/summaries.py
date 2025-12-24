from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from .openai_api import request_chat
from .util import append_jsonl, batched, clamp_text, read_jsonl, write_json


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            candidate = parts[1].lstrip()
            if candidate.startswith("json"):
                candidate = candidate[4:].lstrip()
            text = candidate.strip()
    return json.loads(text)


def build_messages(items: list[dict[str, Any]], *, stage: str) -> list[dict[str, Any]]:
    system = {
        "role": "system",
        "content": (
            "You summarize dense visual captions from a video. "
            "Stay grounded in the captions; do not invent details. "
            "Do not identify real people or guess names. "
            "Write a concise global scene summary focusing on major locations, entities, and activities."
        ),
    }
    user = {
        "role": "user",
        "content": (
            f"Stage: {stage}\n"
            "Summarize the following caption items.\n\n"
            f"{json.dumps(items, ensure_ascii=False)}\n\n"
            "Output JSON ONLY:\n"
            '{ "summary": "..." }\n'
        ),
    }
    return [system, user]


def summarize_captions(
    *,
    api_key: str,
    evidence_jsonl: Path,
    out_json: Path,
    errors_jsonl: Path,
    model: str = "gpt-4o-mini",
    chunk_windows: int = 20,
    max_tokens: int = 1000,
    temperature: float = 0.2,
    timeout_s: float = 60.0,
    retries: int = 3,
    overwrite: bool = False,
    progress_cb: Callable[[str], None] | None = None,
) -> None:
    if out_json.exists() and not overwrite:
        if progress_cb:
            progress_cb("Global summary already exists; skipping.")
        return

    rows = read_jsonl(evidence_jsonl)
    items: list[dict[str, Any]] = []
    for r in rows:
        if isinstance(r.get("_error"), dict):
            continue
        w = r.get("window") or {}
        cap = r.get("caption", "")
        if not isinstance(cap, str) or not cap.strip():
            continue
        items.append(
            {
                "start_s": float(w.get("start_s", 0.0)),
                "end_s": float(w.get("end_s", 0.0)),
                "caption": clamp_text(cap.strip(), 600),
            }
        )
    if not items:
        raise RuntimeError("no captions found in evidence.jsonl")

    chunks = list(batched(items, max(1, int(chunk_windows))))
    if progress_cb:
        progress_cb(f"captions: {len(items)} (chunks: {len(chunks)} × {int(chunk_windows)})")
    chunk_summaries: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        if progress_cb:
            progress_cb(f"chunk {idx}/{len(chunks)}")
        messages = build_messages(chunk, stage=f"chunk {idx}/{len(chunks)}")
        text = request_chat(
            api_key=api_key,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_s=timeout_s,
            retries=retries,
            json_object=True,
        )
        try:
            parsed = extract_json(text)
            summary = parsed.get("summary", "")
            if not isinstance(summary, str) or not summary.strip():
                raise ValueError("missing summary")
            chunk_summaries.append(summary.strip())
        except Exception as e:
            append_jsonl(
                errors_jsonl,
                {
                    "kind": "global_summary_chunk_parse_error",
                    "chunk_index": idx,
                    "chunk_total": len(chunks),
                    "model": model,
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(limit=2),
                    "raw": clamp_text(text, 20000),
                    "_meta": {"generated_at_unix": int(time.time())},
                },
            )
            chunk_summaries.append(clamp_text(text, 2000))

    if len(chunk_summaries) == 1:
        final = chunk_summaries[0]
    else:
        if progress_cb:
            progress_cb("final pass (summary of summaries)")
        messages = build_messages(
            [{"summary": clamp_text(s, 800)} for s in chunk_summaries],
            stage="final",
        )
        text = request_chat(
            api_key=api_key,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_s=timeout_s,
            retries=retries,
            json_object=True,
        )
        try:
            parsed = extract_json(text)
            summary = parsed.get("summary", "")
            if not isinstance(summary, str) or not summary.strip():
                raise ValueError("missing summary")
            final = summary.strip()
        except Exception as e:
            append_jsonl(
                errors_jsonl,
                {
                    "kind": "global_summary_final_parse_error",
                    "model": model,
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(limit=2),
                    "raw": clamp_text(text, 20000),
                    "_meta": {"generated_at_unix": int(time.time())},
                },
            )
            final = clamp_text(text, 4000)

    write_json(
        out_json,
        {
            "model": model,
            "created_at_unix": int(time.time()),
            "caption_count": len(items),
            "chunk_windows": int(chunk_windows),
            "summary": final,
        },
    )
