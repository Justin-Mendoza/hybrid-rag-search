"""Explicit, checksum-pinned local tokenizer provisioning."""

import hashlib
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cohere
from tokenizers import Tokenizer

from hybrid_rag_search.config import Settings


@dataclass(frozen=True)
class TokenizerAsset:
    model: str
    revision: str
    sha256: str


COHERE_EMBED_ENGLISH_LIGHT_V3 = TokenizerAsset(
    model="embed-english-light-v3.0",
    revision="af99964c9f9b5356e7c690bec577743df457fcb2",
    sha256="243655762254d2a961c98b2b8b9d5783dfdb66ff0be4dee94207fb94b35938a0",
)
SUPPORTED_ASSETS = {COHERE_EMBED_ENGLISH_LIGHT_V3.model: COHERE_EMBED_ENGLISH_LIGHT_V3}


class TokenizerFetcher(Protocol):
    def fetch_tokenizer(self, *, model: str) -> Tokenizer: ...


class TokenizerSetupError(RuntimeError):
    """Tokenizer provisioning failed without exposing credentials."""


class TokenizerLoadError(RuntimeError):
    """A pinned local tokenizer could not be safely loaded."""


@dataclass(frozen=True)
class TokenizerProvisionResult:
    path: Path
    sha256: str
    downloaded: bool


@dataclass(frozen=True)
class TokenSpan:
    """One model token and its half-open character range in the input text."""

    token_id: int
    start_char: int
    end_char: int


class TextTokenizer(Protocol):
    """Minimal tokenizer boundary used by deterministic chunking."""

    def tokenize(self, text: str) -> tuple[TokenSpan, ...]: ...


class LocalTextTokenizer:
    """Use a loaded Hugging Face tokenizer without any network fallback."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tokenizer = tokenizer

    def tokenize(self, text: str) -> tuple[TokenSpan, ...]:
        if not isinstance(text, str):
            raise ValueError("Tokenizer input must be a string")
        encoding = self._tokenizer.encode(text, add_special_tokens=False)
        spans = tuple(
            TokenSpan(token_id, start_char, end_char)
            for token_id, (start_char, end_char) in zip(encoding.ids, encoding.offsets, strict=True)
        )
        if any(span.end_char <= span.start_char for span in spans):
            raise TokenizerLoadError("Local tokenizer produced an unusable character offset")
        return spans


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_path(root: Path, asset: TokenizerAsset) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", asset.model):
        raise TokenizerSetupError("Tokenizer model name is not safe for a local filename")
    return root / f"{asset.model}.json"


def load_local_tokenizer(*, root: Path, asset: TokenizerAsset) -> LocalTextTokenizer:
    """Verify and load the pinned asset without contacting Cohere."""

    path = tokenizer_path(root, asset)
    if not path.is_file():
        raise TokenizerLoadError("Local tokenizer is missing; run `make tokenizer-setup`")
    if file_sha256(path) != asset.sha256:
        raise TokenizerLoadError(
            "Local tokenizer checksum does not match the pinned model revision"
        )
    try:
        return LocalTextTokenizer(Tokenizer.from_file(str(path)))
    except Exception:
        raise TokenizerLoadError("Local tokenizer file could not be loaded") from None


def _verified_existing(path: Path, asset: TokenizerAsset) -> TokenizerProvisionResult | None:
    if not path.exists():
        return None
    checksum = file_sha256(path)
    if checksum != asset.sha256:
        raise TokenizerSetupError(
            "Existing tokenizer checksum does not match the pinned model revision"
        )
    return TokenizerProvisionResult(path=path, sha256=checksum, downloaded=False)


def provision_tokenizer(
    *, root: Path, asset: TokenizerAsset, fetcher: TokenizerFetcher
) -> TokenizerProvisionResult:
    """Fetch once through the SDK, verify, and publish without overwriting."""

    path = tokenizer_path(root, asset)
    existing = _verified_existing(path, asset)
    if existing is not None:
        return existing
    root.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        tokenizer = fetcher.fetch_tokenizer(model=asset.model)
        serialized = tokenizer.to_str(pretty=True).encode()
        checksum = hashlib.sha256(serialized).hexdigest()
        if checksum != asset.sha256:
            raise TokenizerSetupError(
                "Downloaded tokenizer checksum does not match the pinned model revision"
            )
        with tempfile.NamedTemporaryFile(dir=root, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(serialized)
            temporary.flush()
            os.fsync(temporary.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            raced = _verified_existing(path, asset)
            if raced is None:  # pragma: no cover - path existed in the exception branch
                raise TokenizerSetupError("Tokenizer setup race produced no local asset") from None
            return raced
        return TokenizerProvisionResult(path=path, sha256=checksum, downloaded=True)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def configured_tokenizer_asset(settings: Settings) -> TokenizerAsset:
    try:
        return SUPPORTED_ASSETS[settings.cohere_embed_model]
    except KeyError:
        raise TokenizerSetupError(
            "No pinned tokenizer is configured for the selected embedding model"
        ) from None


def provision_configured_tokenizer(settings: Settings) -> TokenizerProvisionResult:
    asset = configured_tokenizer_asset(settings)
    existing = _verified_existing(tokenizer_path(settings.tokenizer_root, asset), asset)
    if existing is not None:
        return existing
    if settings.cohere_api_key is None:
        raise TokenizerSetupError("A Cohere API key is required for tokenizer setup")
    try:
        client = cohere.ClientV2(api_key=settings.cohere_api_key.get_secret_value())
        return provision_tokenizer(root=settings.tokenizer_root, asset=asset, fetcher=client)
    except TokenizerSetupError:
        raise
    except Exception:
        raise TokenizerSetupError("Cohere SDK could not fetch the tokenizer") from None


def load_configured_tokenizer(settings: Settings) -> LocalTextTokenizer:
    """Load the tokenizer selected by application settings from local storage."""

    try:
        asset = configured_tokenizer_asset(settings)
    except TokenizerSetupError as error:
        raise TokenizerLoadError(str(error)) from None
    return load_local_tokenizer(root=settings.tokenizer_root, asset=asset)
