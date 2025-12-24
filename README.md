# A simple Video RAG 

A very simple Entity based Video Framework for Questions Answering over videos.

# Methodology

We follow a very simple bare bones methodology to build a Video RAG. Our current approach uses a very simple VLM to extract evidence from the video and then use a simple Retrieval to answer questions over the video.

Framework follows the following steps:
1. Extract Frames from the video (using ffmpeg)
2. Extract Evidence from the video (using a VLM) - defaults to GPT-4o-mini
Evidence Extraction is done in a sliding window manner, entities are tracked across windows and new entities are added to the entity roster, simple relationships are extracted between entities.
3. We extract a global summary from the video (using a VLM) - defaults to GPT-4o-mini, to help follow global context and global scene understanding/question answering. 
4. We build an entity vector index from the evidence (using a VLM) - defaults to text-embedding-3-small
5. Based on questions at query time, we retrieve the most relevant entity based evidence from the vector index and use them to answer the question.


## Setup

- Put your OpenAI key in `.env` (this repo ignores it):
  - `OPENAI_API_KEY=...`
- Install deps:
  - `uv sync`

## Quickstart (Python API)

See `example_videorag.py` for a runnable example.

```python
from videorag import VideoRAG

rag = VideoRAG("IMG_0617.mov", fps=1)
rag.index()  # resumes by default; overwrite=True rebuilds

print(rag.answer("What is the humanoid robot doing?", top_k=5))
```

## What `index()` produces (per-run)

Runs are isolated under `artifacts/runs/<video_hash>_fps<fps>/`:

- `frames/` extracted images + `frames/index.jsonl`
- `evidence/evidence.jsonl` dense captions + entities + relations (timestamped)
- `evidence/state.json` rolling state (entity roster + rolling summary state)
- `evidence/errors.jsonl` raw model outputs when parsing fails
- `summaries/global_summary.json` captions-only global scene summary
- `vector/entities.npz` + `vector/entities.json` entity embedding index + timestamped mentions

## Retrieval

```python
ctx = rag.retrieve("robot arm", top_k=5, max_mentions=6)
# ctx includes ctx["global_summary"] + ctx["results"] (timestamped local evidence)
```

# Improvements

- [ ] Temporal reasoning currentlly not supported. I think this should be easy enough to achieve with a simple routing mechanism, since if we know it is a temporal call, approach could follow a very simple termporal routing mechanism. (Based on timestamps of the evidence)
- [] The evidence creation step is super slow, imo we should be able to use something like CLIP to pick only the most relevant frames for the evidence creation step.
- [] The entity graph is also super simple, we should be able to handle more complex QA with a slightly more involved graph structure. 
- [] Hybrid retreval with a sparse retriever (BM25 like) is super easy to implement and is guaranteed to improve performance. 

## Notes

All pipeline functionality lives under `videorag/`.
