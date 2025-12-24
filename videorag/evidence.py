from __future__ import annotations

import dataclasses
import json
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Iterable

from .openai_api import encode_image_data_url, request_chat
from .util import append_jsonl, clamp_text, format_progress, read_json, read_jsonl, write_json


@dataclasses.dataclass(frozen=True)
class Frame:
    frame_index: int
    timestamp_s: float
    timestamp: str
    path: str


def load_frames(index_path: Path) -> list[Frame]:
    rows = read_jsonl(index_path)
    frames: list[Frame] = []
    for row in rows:
        p = str(row["path"])
        # Support both absolute paths and paths relative to the index location.
        if not Path(p).is_absolute():
            p = str((index_path.parent / p).resolve())
        frames.append(
            Frame(
                frame_index=int(row["frame_index"]),
                timestamp_s=float(row["timestamp_s"]),
                timestamp=str(row["timestamp"]),
                path=p,
            )
        )
    return frames


def iter_windows(
    frames: list[Frame],
    window_seconds: float,
    stride_seconds: float,
) -> Iterable[tuple[float, float, list[Frame]]]:
    if not frames:
        return
    start_s = frames[0].timestamp_s
    end_s = frames[-1].timestamp_s + 1e-6
    cur = start_s
    frame_i = 0
    while cur < end_s:
        w_start = cur
        w_end = cur + window_seconds
        window_frames: list[Frame] = []
        while frame_i < len(frames) and frames[frame_i].timestamp_s < w_end:
            if frames[frame_i].timestamp_s >= w_start:
                window_frames.append(frames[frame_i])
            frame_i += 1

        yield (w_start, w_end, window_frames)

        cur += stride_seconds
        while frame_i > 0 and frame_i < len(frames) and frames[frame_i - 1].timestamp_s >= cur:
            frame_i -= 1


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


def next_entity_id_hint(roster: Any) -> str:
    if not isinstance(roster, list):
        return "E1"
    max_n = 0
    for e in roster:
        if not isinstance(e, dict):
            continue
        eid = e.get("id")
        if isinstance(eid, str) and eid.startswith("E"):
            tail = eid[1:]
            if tail.isdigit():
                max_n = max(max_n, int(tail))
    return f"E{max_n + 1}"


def default_state() -> dict[str, Any]:
    return {
        "entity_roster": [],
        "rolling_summary": "",
        "chunk_buffer": [],
        "next_summary_at_s": 0.0,
        "last_present": [],
    }


def load_state(state_path: Path) -> dict[str, Any]:
    if not state_path.exists():
        return default_state()
    st = read_json(state_path)
    if not isinstance(st, dict):
        return default_state()
    st.setdefault("entity_roster", [])
    st.setdefault("rolling_summary", "")
    st.setdefault("chunk_buffer", [])
    st.setdefault("next_summary_at_s", 0.0)
    st.setdefault("last_present", [])
    return st


def build_window_messages(
    *,
    w_start: float,
    w_end: float,
    window_frames: list[Frame],
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    roster = state.get("entity_roster", [])
    rolling_summary = state.get("rolling_summary", "")
    next_id = next_entity_id_hint(roster)

    system = {
        "role": "system",
        "content": (
            "You analyze short sequences of video frames. "
            "You must stay grounded in what is visible. "
            "Do not identify real people or guess names/identities; describe people anonymously. "
            "Extract general entities (people, objects, screens, text, locations) and relations between them. "
            "Use stable entity IDs like E1, E2 based on the provided roster."
        ),
    }

    frame_list = "\n".join(f"- {f.path} @ {f.timestamp} ({f.timestamp_s:.3f}s)" for f in window_frames)
    user_text = f"""
Time window: [{w_start:.3f}, {w_end:.3f}) seconds

Existing entity roster (may be empty):
{json.dumps(roster, ensure_ascii=False)}

Rolling summary so far (may be empty):
{clamp_text(str(rolling_summary), 1500)}

Frames in this window (ordered):
{frame_list if frame_list else "(no frames)"}

Task:
1) Write a dense, grounded caption describing what is visible across the frames in this time window.
2) Identify which existing roster entities appear in these frames.
3) Add any new salient entities (people/objects/screens/text/locations) with a short grounded descriptor.
4) Extract grounded relations/events between entities (e.g., looks_at, holds, uses, walks_past, speaks_to (inferred)).

New entity IDs must start at: {next_id}

Rules (important):
- You MUST stay grounded in what is visible in the provided frames.
- You MUST NOT mention any entity ID unless it appears in the provided roster OR you include it in new_entities in this same output.
- If the roster is empty, introduce any salient entities you reference (start with E1, E2, ...).
- Do not invent on-screen text: only include text you can read.
- If a relation is inferred (e.g., speaks_to without audio), include it but lower confidence and explain the visual cues.

Output JSON ONLY with this schema:
{{
  "window": {{"start_s": number, "end_s": number}},
  "keyframes": ["path1", "path2"],
  "caption": "dense grounded description",
  "entities_present": [{{"id": "E1", "confidence": 0.0-1.0}}],
  "new_entities": [{{"id": "E3", "type": "person|object|screen|text|location|other", "descriptor": "..."}}],
  "relations": [
    {{
      "type": "speaks_to|looks_at|holds|uses|moves|gesture|scene_change|other",
      "subject": "E1|unknown",
      "object": "E2|unknown",
      "confidence": 0.0-1.0,
      "evidence": ["frame_path1", "frame_path2"],
      "notes": "short, grounded"
    }}
  ],
  "on_screen_text": ["verbatim snippets"],
  "uncertainties": ["things that are unclear"],
  "confidence": 0.0-1.0
}}
""".strip()

    content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    for fr in window_frames:
        content.append({"type": "image_url", "image_url": {"url": encode_image_data_url(Path(fr.path))}})
    user = {"role": "user", "content": content}
    return [system, user]


def build_summary_messages(*, rolling_summary: str, chunk_windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    system = {
        "role": "system",
        "content": (
            "You summarize timestamped video-window logs into a concise rolling summary. "
            "Stay grounded in the provided window captions/relations. "
            "Do not invent entities or rename entity IDs; preserve IDs like E1, E2 exactly. "
            "You MAY incorporate new entity IDs if they appear in the provided chunk windows (e.g., in new_entities). "
            "Be concise, but keep relevant entity continuity and key relations."
        ),
    }
    user = {
        "role": "user",
        "content": (
            "Update the rolling summary using the newest chunk.\n\n"
            f"Previous rolling summary (may be empty):\n{clamp_text(rolling_summary, 2500)}\n\n"
            "New chunk windows (JSON):\n"
            f"{json.dumps(chunk_windows, ensure_ascii=False)}\n\n"
            "Output a concise summary as PLAIN TEXT (no JSON, no code fences).\n"
            "Length constraints (important):\n"
            "- Target <= 120 words total.\n"
            "- Hard cap <= 900 characters.\n"
       ),
    }
    return [system, user]


def safe_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return default


def generate_evidence(
    *,
    api_key: str,
    index_path: Path,
    evidence_path: Path,
    state_path: Path,
    errors_path: Path,
    model: str = "gpt-4o-mini",
    window_s: float = 2.0,
    stride_s: float = 2.0,
    max_tokens: int = 900,
    temperature: float = 0.2,
    timeout_s: float = 60.0,
    retries: int = 3,
    summary_interval_s: float = 10.0,
    summary_model: str | None = None,
    summary_max_tokens: int = 350,
    summary_temperature: float = 0.2,
    finalize_summary: bool = True,
    overwrite: bool = False,
    progress_cb: Callable[[str], None] | None = None,
    progress_every: int = 5,
) -> None:
    frames = load_frames(index_path)
    windows = list(iter_windows(frames, window_s, stride_s))

    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite:
        if evidence_path.exists():
            evidence_path.unlink()
        if state_path.exists():
            state_path.unlink()

    state = load_state(state_path)
    if summary_interval_s > 0 and safe_float(state.get("next_summary_at_s"), 0.0) <= 0.0:
        state["next_summary_at_s"] = float(summary_interval_s)

    existing = 0
    if evidence_path.exists():
        with evidence_path.open("r", encoding="utf-8") as f:
            for _ in f:
                existing += 1

    windows_to_do = windows[existing:]
    if not windows_to_do:
        if progress_cb:
            progress_cb("Evidence already complete; nothing to do.")
        return

    sum_model = summary_model or model
    total = len(windows)
    done = existing
    if progress_cb:
        progress_cb(f"windows: {total} (already have {existing}, remaining {len(windows_to_do)})")

    with evidence_path.open("a", encoding="utf-8") as out_f:
        for (w_start, w_end, window_frames) in windows_to_do:
            done += 1
            if progress_cb and (done == total or (done % max(1, progress_every) == 0)):
                progress_cb(f"{format_progress(done, total)}  current=[{w_start:.1f},{w_end:.1f})s frames={len(window_frames)}")
            messages = build_window_messages(
                w_start=w_start, w_end=w_end, window_frames=window_frames, state=state
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
            except Exception as e:
                append_jsonl(
                    errors_path,
                    {
                        "kind": "window_parse_error",
                        "window": {"start_s": w_start, "end_s": w_end},
                        "keyframes": [fr.path for fr in window_frames],
                        "model": model,
                        "error": f"{type(e).__name__}: {e}",
                        "traceback": traceback.format_exc(limit=2),
                        "raw": clamp_text(text, 20000),
                        "_meta": {"generated_at_unix": int(time.time())},
                    },
                )
                placeholder = {
                    "window": {"start_s": w_start, "end_s": w_end},
                    "keyframes": [fr.path for fr in window_frames],
                    "_error": {"kind": "window_parse_error", "errors_path": str(errors_path)},
                    "_meta": {"model": model, "generated_at_unix": int(time.time())},
                }
                out_f.write(json.dumps(placeholder, ensure_ascii=False) + "\n")
                out_f.flush()
                if progress_cb:
                    progress_cb(f"parse error in window [{w_start:.1f},{w_end:.1f})s (logged to errors.jsonl)")
                continue

            parsed.setdefault("window", {"start_s": w_start, "end_s": w_end})
            parsed.setdefault("keyframes", [fr.path for fr in window_frames])
            parsed["_meta"] = {"model": model, "generated_at_unix": int(time.time())}
            out_f.write(json.dumps(parsed, ensure_ascii=False) + "\n")
            out_f.flush()

            if isinstance(parsed.get("_error"), dict):
                continue

            new_entities = parsed.get("new_entities", [])
            present = parsed.get("entities_present", [])
            chunk_buffer = state.get("chunk_buffer", [])
            if not isinstance(chunk_buffer, list):
                chunk_buffer = []

            if new_entities:
                roster = list(state.get("entity_roster", []))
                known = {e.get("id") for e in roster if isinstance(e, dict)}
                for e in new_entities:
                    if isinstance(e, dict) and e.get("id") not in known:
                        roster.append(e)
                        known.add(e.get("id"))
                state["entity_roster"] = roster

            roster = list(state.get("entity_roster", []))
            known = {e.get("id") for e in roster if isinstance(e, dict)}
            referenced: set[str] = set()
            for p in present or []:
                if isinstance(p, dict) and isinstance(p.get("id"), str):
                    referenced.add(p["id"])
            for rel in parsed.get("relations") or []:
                if isinstance(rel, dict):
                    for k in ("subject", "object"):
                        v = rel.get(k)
                        if isinstance(v, str) and v != "unknown":
                            referenced.add(v)
            for rid in sorted(referenced):
                if rid not in known:
                    roster.append({"id": rid, "descriptor": "unknown (auto-added; rerun recommended)"})
                    known.add(rid)
            state["entity_roster"] = roster
            state["last_present"] = present

            chunk_buffer.append(
                {
                    "window": parsed.get("window"),
                    "caption": parsed.get("caption", ""),
                    "entities_present": parsed.get("entities_present", []),
                    "new_entities": parsed.get("new_entities", []),
                    "relations": parsed.get("relations", []),
                    "on_screen_text": parsed.get("on_screen_text", []),
                }
            )
            state["chunk_buffer"] = chunk_buffer

            if summary_interval_s > 0:
                next_at = safe_float(state.get("next_summary_at_s"), float(summary_interval_s))
                if w_end + 1e-6 >= next_at and chunk_buffer:
                    if progress_cb:
                        progress_cb(f"updating rolling summary at t≈{w_end:.1f}s (interval {summary_interval_s:g}s)")
                    sum_messages = build_summary_messages(
                        rolling_summary=str(state.get("rolling_summary", "")),
                        chunk_windows=chunk_buffer,
                    )
                    sum_text = request_chat(
                        api_key=api_key,
                        model=sum_model,
                        messages=sum_messages,
                        max_tokens=summary_max_tokens,
                        temperature=summary_temperature,
                        timeout_s=timeout_s,
                        retries=retries,
                        json_object=False,
                    )
                    if isinstance(sum_text, str) and sum_text.strip():
                        state["rolling_summary"] = sum_text.strip()
                    state["chunk_buffer"] = []
                    while next_at <= w_end + 1e-6:
                        next_at += float(summary_interval_s)
                    state["next_summary_at_s"] = next_at

            write_json(state_path, state)

    if summary_interval_s > 0 and finalize_summary:
        state = load_state(state_path)
        chunk_buffer = state.get("chunk_buffer", [])
        if isinstance(chunk_buffer, list) and chunk_buffer:
            if progress_cb:
                progress_cb("finalizing rolling summary for last partial chunk")
            sum_messages = build_summary_messages(
                rolling_summary=str(state.get("rolling_summary", "")),
                chunk_windows=chunk_buffer,
            )
            sum_text = request_chat(
                api_key=api_key,
                model=sum_model,
                messages=sum_messages,
                max_tokens=summary_max_tokens,
                temperature=summary_temperature,
                timeout_s=timeout_s,
                retries=retries,
                json_object=False,
            )
            if isinstance(sum_text, str) and sum_text.strip():
                state["rolling_summary"] = sum_text.strip()
            state["chunk_buffer"] = []
            write_json(state_path, state)
