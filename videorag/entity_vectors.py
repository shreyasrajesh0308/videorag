from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .openai_api import request_embeddings, request_chat
from .util import clamp_text, format_progress, read_json, read_jsonl, write_json


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def entity_ids_in_window(rec: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for p in rec.get("entities_present") or []:
        if isinstance(p, dict) and isinstance(p.get("id"), str):
            ids.add(p["id"])
    for e in rec.get("new_entities") or []:
        if isinstance(e, dict) and isinstance(e.get("id"), str):
            ids.add(e["id"])
    for r in rec.get("relations") or []:
        if isinstance(r, dict):
            for k in ("subject", "object"):
                v = r.get(k)
                if isinstance(v, str) and v != "unknown":
                    ids.add(v)
    return ids


def build_entity_docs(
    *,
    evidence: list[dict[str, Any]],
    roster: list[dict[str, Any]],
    max_mentions: int,
    max_chars: int,
) -> list[dict[str, Any]]:
    roster_by_id: dict[str, dict[str, Any]] = {}
    for e in roster:
        if isinstance(e, dict) and isinstance(e.get("id"), str):
            roster_by_id[e["id"]] = e

    entity_mentions: dict[str, list[dict[str, Any]]] = {}
    entity_new: dict[str, dict[str, Any]] = {}

    for rec in evidence:
        if isinstance(rec.get("_error"), dict):
            continue
        for e in rec.get("new_entities") or []:
            if isinstance(e, dict) and isinstance(e.get("id"), str):
                entity_new.setdefault(e["id"], e)

        ids = entity_ids_in_window(rec)
        w = rec.get("window") or {}
        start_s = float(w.get("start_s", 0.0))
        end_s = float(w.get("end_s", 0.0))
        caption = rec.get("caption", "")
        if not isinstance(caption, str):
            caption = ""
        keyframes = rec.get("keyframes") or []
        if not isinstance(keyframes, list):
            keyframes = []

        for eid in ids:
            rels: list[dict[str, Any]] = []
            for r in rec.get("relations") or []:
                if not isinstance(r, dict):
                    continue
                if r.get("subject") == eid or r.get("object") == eid:
                    rels.append(r)

            entity_mentions.setdefault(eid, []).append(
                {
                    "start_s": start_s,
                    "end_s": end_s,
                    "caption": caption,
                    "keyframes": [x for x in keyframes if isinstance(x, str)],
                    "relations": rels,
                    "on_screen_text": rec.get("on_screen_text") if isinstance(rec.get("on_screen_text"), list) else [],
                    "uncertainties": rec.get("uncertainties") if isinstance(rec.get("uncertainties"), list) else [],
                }
            )

    docs: list[dict[str, Any]] = []
    for eid, mentions in sorted(entity_mentions.items()):
        base = roster_by_id.get(eid) or entity_new.get(eid) or {"id": eid}
        etype = base.get("type") if isinstance(base.get("type"), str) else "other"
        desc = base.get("descriptor") if isinstance(base.get("descriptor"), str) else ""

        mentions_sorted = sorted(mentions, key=lambda m: (float(m.get("start_s", 0.0)), float(m.get("end_s", 0.0))))
        mentions_sorted = mentions_sorted[:max_mentions]

        parts: list[str] = [f"{eid} ({etype})", desc]
        for m in mentions_sorted[: min(mentions_sorted.__len__(), 25)]:
            cap = m.get("caption", "")
            if isinstance(cap, str) and cap.strip():
                parts.append(clamp_text(cap.strip(), 280))
            for r in m.get("relations") or []:
                if not isinstance(r, dict):
                    continue
                t = r.get("type")
                s = r.get("subject")
                o = r.get("object")
                notes = r.get("notes")
                bits = [x for x in (t, s, o, notes) if isinstance(x, str) and x.strip()]
                if bits:
                    parts.append(clamp_text(" ".join(bits), 240))
        embed_text = clamp_text("\n".join([p for p in parts if p]), max_chars)

        docs.append(
            {
                "entity_id": eid,
                "type": etype,
                "descriptor": desc,
                "mentions": mentions_sorted,
                "embed_text": embed_text,
                "embed_text_hash": stable_hash(embed_text),
            }
        )
    return docs


def build_entity_vector_index(
    *,
    api_key: str,
    evidence_jsonl: Path,
    state_json: Path,
    out_npz: Path,
    out_meta_json: Path,
    embedding_model: str = "text-embedding-3-small",
    timeout_s: float = 60.0,
    retries: int = 3,
    max_mentions: int = 60,
    max_chars: int = 6000,
    overwrite: bool = False,
    progress_cb: Any = None,
) -> None:
    if out_npz.exists() and out_meta_json.exists() and not overwrite:
        return

    evidence = read_jsonl(evidence_jsonl)
    state: dict[str, Any] = read_json(state_json) if state_json.exists() else {}
    roster = state.get("entity_roster", [])
    if not isinstance(roster, list):
        roster = []

    docs = build_entity_docs(
        evidence=evidence,
        roster=[e for e in roster if isinstance(e, dict)],
        max_mentions=max_mentions,
        max_chars=max_chars,
    )
    if not docs:
        raise RuntimeError("no entities found to index")

    vectors: list[list[float]] = []
    total = len(docs)
    for i, d in enumerate(docs, start=1):
        emb = request_embeddings(
            api_key=api_key,
            model=embedding_model,
            text=d["embed_text"],
            timeout_s=timeout_s,
            retries=retries,
        )
        vectors.append(emb)
        if progress_cb and (i == total or i % 10 == 0):
            progress_cb(format_progress(i, total) + "  entities embedded")

    mat = np.array(vectors, dtype=np.float32)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, vectors=mat)
    write_json(
        out_meta_json,
        {
            "embedding_model": embedding_model,
            "created_at_unix": int(time.time()),
            "entities": docs,
        },
    )


def format_time(s: float) -> str:
    m = int(s // 60)
    sec = s - 60 * m
    return f"{m:02d}:{sec:06.3f}"


def retrieve_entities(
    *,
    api_key: str,
    vectors_npz: Path,
    meta_json: Path,
    query: str,
    top_k: int = 5,
    max_mentions: int = 6,
    embedding_model: str = "text-embedding-3-small",
    timeout_s: float = 60.0,
    retries: int = 3,
) -> dict[str, Any]:
    meta = read_json(meta_json)
    entities: list[dict[str, Any]] = meta.get("entities", [])
    if not isinstance(entities, list) or not entities:
        raise RuntimeError("meta contains no entities")

    mat = np.load(vectors_npz)["vectors"].astype(np.float32)
    if mat.shape[0] != len(entities):
        raise RuntimeError("vectors/meta mismatch")

    q_emb = request_embeddings(
        api_key=api_key,
        model=embedding_model,
        text=query,
        timeout_s=timeout_s,
        retries=retries,
    )
    q = np.array(q_emb, dtype=np.float32)
    mat_norm = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
    q_norm = q / (np.linalg.norm(q) + 1e-12)
    sims = mat_norm @ q_norm

    k = max(1, int(top_k))
    top_idx = np.argsort(-sims)[:k]

    results: list[dict[str, Any]] = []
    for i in top_idx.tolist():
        e = entities[int(i)]
        results.append(
            {
                "entity_id": e.get("entity_id"),
                "type": e.get("type", "other"),
                "descriptor": e.get("descriptor", ""),
                "score": float(sims[int(i)]),
                "mentions": (e.get("mentions", []) or [])[: max(0, int(max_mentions))],
            }
        )

    return {
        "query": query,
        "top_k": k,
        "embedding_model": embedding_model,
        "results": results,
    }


def answer_with_context(
    *,
    api_key: str,
    query: str,
    global_summary: str,
    local_context: str,
    answer_model: str = "gpt-4o-mini",
    max_tokens: int = 400,
    temperature: float = 0.2,
    timeout_s: float = 60.0,
    retries: int = 3,
) -> str:
    system = {
        "role": "system",
        "content": (
            "You answer questions about a video using provided context. "
            "Use timestamp ranges as evidence; do not invent events not supported by the context. "
            "If uncertain, say so. "
            "Do not identify real people by name."
        ),
    }
    parts = []
    if global_summary.strip():
        parts.append("Global scene understanding (high-level, may be incomplete):\n" + global_summary.strip())
    parts.append(local_context.strip())
    user = {
        "role": "user",
        "content": (
            f"Question: {query}\n\n"
            + "\n\n".join(parts)
            + "\n\nAnswer in a short paragraph, then include an 'Evidence' list with 1-5 bullets citing the exact "
            "timestamp range(s) (e.g., 00:28.000–00:30.000) and the relevant entity IDs."
        ),
    }
    return request_chat(
        api_key=api_key,
        model=answer_model,
        messages=[system, user],
        max_tokens=max_tokens,
        temperature=temperature,
        timeout_s=timeout_s,
        retries=retries,
        json_object=False,
    ).strip()
