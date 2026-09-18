"""Command-line entry point for explicit local tokenizer provisioning."""

from hybrid_rag_search.config import get_settings
from hybrid_rag_search.tokenization import TokenizerSetupError, provision_configured_tokenizer


def main() -> int:
    try:
        result = provision_configured_tokenizer(get_settings())
    except TokenizerSetupError as error:
        print(f"Tokenizer setup failed: {error}")
        return 1
    action = "downloaded" if result.downloaded else "already verified"
    print(f"Tokenizer {action}: {result.path} (sha256={result.sha256})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
