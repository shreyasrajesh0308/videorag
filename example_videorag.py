from videorag import VideoRAG


def main() -> None:
    rag = VideoRAG("IMG_0617.mov", fps=0.5)
    rag.index()  # resumes by default; pass overwrite=True to rebuild

    query = "What was the person on the phone wearing"
    ctx = rag.retrieve(query, top_k=5)
    print("Global summary:")
    print(ctx.get("global_summary", ""))
    print("\nAnswer:")
    print(rag.answer(query, top_k=5))


if __name__ == "__main__":
    main()
