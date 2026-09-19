"""Manual recovery command for durable ingestion jobs."""

import asyncio

from hybrid_rag_search.config import get_settings
from hybrid_rag_search.ingestion.runtime import run_configured_recovery
from hybrid_rag_search.worker import ingest_document


def main() -> None:
    result = asyncio.run(run_configured_recovery(get_settings(), ingest_document))
    print(
        "ingestion recovery: "
        f"requeued={result.requeued_expired} "
        f"failed={result.failed_expired} "
        f"published={result.published}"
    )


if __name__ == "__main__":
    main()
