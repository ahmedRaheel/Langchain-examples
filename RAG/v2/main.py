from pathlib import Path

from rag import (
    build_rag,
    retrieve
)

from generator import (
    generate_answer
)


DOCUMENT_PATH = (
    Path("document")
    / "python.txt"
)


def main():

    text = DOCUMENT_PATH.read_text(
        encoding="utf-8"
    )

    print("Building RAG knowledge base...")

    build_rag(
        text,
        DOCUMENT_PATH.name
    )

    print(
        "\nAdvanced RAG is ready."
    )

    while True:

        query = input(
            "\nYou : "
        ).strip()

        if query.lower() in {
            "exit",
            "quit"
        }:
            break

        results = retrieve(
            query,
            top_k=5
        )

        print(
            "\nRetrieved Sources Ranking:"
        )

        for result in results:

            print(
                f"- "
                f"{result['source']} "
                f"(chunk "
                f"{result['chunk_id']})"
            )

        answer = generate_answer(
            query,
            results
        )

        print(
            "\nRAG :"
        )

        print(answer)


if __name__ == "__main__":
    main()


