import hashlib
from pathlib import Path
from unittest.mock import Mock

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from hybrid_rag_search.config import Settings
from hybrid_rag_search.tokenization import (
    COHERE_EMBED_ENGLISH_LIGHT_V3,
    SUPPORTED_ASSETS,
    LocalTextTokenizer,
    TokenizerAsset,
    TokenizerLoadError,
    TokenizerSetupError,
    TokenSpan,
    configured_tokenizer_asset,
    file_sha256,
    load_configured_tokenizer,
    load_local_tokenizer,
    provision_configured_tokenizer,
    provision_tokenizer,
    tokenizer_path,
)


def fake_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "remote": 1, "work": 2}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return tokenizer


def fake_asset(tokenizer: Tokenizer, model: str = "test-model") -> TokenizerAsset:
    checksum = hashlib.sha256(tokenizer.to_str(pretty=True).encode()).hexdigest()
    return TokenizerAsset(model=model, revision="test-revision", sha256=checksum)


def test_provisions_verified_tokenizer_atomically(tmp_path: Path) -> None:
    tokenizer = fake_tokenizer()
    asset = fake_asset(tokenizer)
    fetcher = Mock(fetch_tokenizer=Mock(return_value=tokenizer))
    result = provision_tokenizer(root=tmp_path, asset=asset, fetcher=fetcher)
    assert result.downloaded is True
    assert result.path == tmp_path / "test-model.json"
    assert result.sha256 == asset.sha256 == file_sha256(result.path)
    assert Tokenizer.from_file(str(result.path)).encode("remote work").ids == [1, 2]
    fetcher.fetch_tokenizer.assert_called_once_with(model="test-model")
    assert tuple(tmp_path.iterdir()) == (result.path,)


def test_existing_verified_asset_makes_no_sdk_call(tmp_path: Path) -> None:
    tokenizer = fake_tokenizer()
    asset = fake_asset(tokenizer)
    path = tokenizer_path(tmp_path, asset)
    path.write_text(tokenizer.to_str(pretty=True))
    fetcher = Mock()
    result = provision_tokenizer(root=tmp_path, asset=asset, fetcher=fetcher)
    assert result.downloaded is False
    assert result.path == path
    fetcher.fetch_tokenizer.assert_not_called()


def test_refuses_existing_asset_with_wrong_checksum(tmp_path: Path) -> None:
    asset = fake_asset(fake_tokenizer())
    path = tokenizer_path(tmp_path, asset)
    path.write_text("tampered")
    with pytest.raises(TokenizerSetupError, match="Existing tokenizer checksum"):
        provision_tokenizer(root=tmp_path, asset=asset, fetcher=Mock())
    assert path.read_text() == "tampered"


def test_refuses_download_with_wrong_checksum_without_publishing(tmp_path: Path) -> None:
    tokenizer = fake_tokenizer()
    asset = TokenizerAsset("test-model", "revision", "0" * 64)
    fetcher = Mock(fetch_tokenizer=Mock(return_value=tokenizer))
    with pytest.raises(TokenizerSetupError, match="Downloaded tokenizer checksum"):
        provision_tokenizer(root=tmp_path, asset=asset, fetcher=fetcher)
    assert tuple(tmp_path.iterdir()) == ()


def test_concurrent_verified_install_wins_without_overwrite(tmp_path: Path, monkeypatch) -> None:
    tokenizer = fake_tokenizer()
    asset = fake_asset(tokenizer)

    def concurrent_link(source: Path, destination: Path) -> None:
        destination.write_bytes(source.read_bytes())
        raise FileExistsError

    monkeypatch.setattr("hybrid_rag_search.tokenization.os.link", concurrent_link)
    result = provision_tokenizer(
        root=tmp_path,
        asset=asset,
        fetcher=Mock(fetch_tokenizer=Mock(return_value=tokenizer)),
    )
    assert result.downloaded is False
    assert file_sha256(result.path) == asset.sha256


def test_rejects_unsafe_model_filename(tmp_path: Path) -> None:
    asset = fake_asset(fake_tokenizer(), "../escape")
    with pytest.raises(TokenizerSetupError, match="not safe"):
        provision_tokenizer(root=tmp_path, asset=asset, fetcher=Mock())


def test_configured_asset_is_pinned_to_selected_model() -> None:
    asset = configured_tokenizer_asset(Settings())
    assert asset == COHERE_EMBED_ENGLISH_LIGHT_V3
    assert asset.revision == "af99964c9f9b5356e7c690bec577743df457fcb2"
    assert asset.sha256 == "243655762254d2a961c98b2b8b9d5783dfdb66ff0be4dee94207fb94b35938a0"


def test_rejects_unpinned_model() -> None:
    with pytest.raises(TokenizerSetupError, match="No pinned tokenizer"):
        configured_tokenizer_asset(Settings(cohere_embed_model="another-model"))


def test_configured_setup_requires_key_only_when_asset_is_missing(tmp_path: Path) -> None:
    settings = Settings(tokenizer_root=tmp_path, cohere_api_key=None)
    with pytest.raises(TokenizerSetupError, match="API key"):
        provision_configured_tokenizer(settings)


def test_configured_setup_reuses_asset_without_key(tmp_path: Path, monkeypatch) -> None:
    tokenizer = fake_tokenizer()
    asset = fake_asset(tokenizer, COHERE_EMBED_ENGLISH_LIGHT_V3.model)
    monkeypatch.setitem(SUPPORTED_ASSETS, asset.model, asset)
    path = tokenizer_path(tmp_path, asset)
    path.write_text(tokenizer.to_str(pretty=True))
    result = provision_configured_tokenizer(Settings(tokenizer_root=tmp_path, cohere_api_key=None))
    assert result.downloaded is False
    assert result.sha256 == asset.sha256


def test_configured_setup_fetches_through_sdk(tmp_path: Path, monkeypatch) -> None:
    tokenizer = fake_tokenizer()
    asset = fake_asset(tokenizer, COHERE_EMBED_ENGLISH_LIGHT_V3.model)
    monkeypatch.setitem(SUPPORTED_ASSETS, asset.model, asset)
    client = Mock(fetch_tokenizer=Mock(return_value=tokenizer))
    client_factory = Mock(return_value=client)
    monkeypatch.setattr("hybrid_rag_search.tokenization.cohere.ClientV2", client_factory)
    settings = Settings(tokenizer_root=tmp_path, cohere_api_key="test-key")
    result = provision_configured_tokenizer(settings)
    assert result.downloaded is True
    client_factory.assert_called_once_with(api_key="test-key")
    client.fetch_tokenizer.assert_called_once_with(model=asset.model)


def test_configured_setup_maps_sdk_failure_to_safe_error(tmp_path: Path, monkeypatch) -> None:
    client = Mock(fetch_tokenizer=Mock(side_effect=RuntimeError("secret provider detail")))
    monkeypatch.setattr("hybrid_rag_search.tokenization.cohere.ClientV2", Mock(return_value=client))
    settings = Settings(tokenizer_root=tmp_path, cohere_api_key="test-key")
    with pytest.raises(TokenizerSetupError, match="could not fetch") as caught:
        provision_configured_tokenizer(settings)
    assert "secret provider detail" not in str(caught.value)


def test_configured_setup_preserves_safe_setup_error(tmp_path: Path, monkeypatch) -> None:
    tokenizer = fake_tokenizer()
    client = Mock(fetch_tokenizer=Mock(return_value=tokenizer))
    monkeypatch.setattr("hybrid_rag_search.tokenization.cohere.ClientV2", Mock(return_value=client))
    settings = Settings(tokenizer_root=tmp_path, cohere_api_key="test-key")
    with pytest.raises(TokenizerSetupError, match="Downloaded tokenizer checksum"):
        provision_configured_tokenizer(settings)


def test_local_tokenizer_returns_model_ids_and_character_offsets(tmp_path: Path) -> None:
    tokenizer = fake_tokenizer()
    asset = fake_asset(tokenizer)
    path = tokenizer_path(tmp_path, asset)
    path.write_text(tokenizer.to_str(pretty=True))

    local = load_local_tokenizer(root=tmp_path, asset=asset)

    assert isinstance(local, LocalTextTokenizer)
    assert local.tokenize("remote work") == (
        TokenSpan(token_id=1, start_char=0, end_char=6),
        TokenSpan(token_id=2, start_char=7, end_char=11),
    )
    assert local.tokenize("") == ()


def test_local_tokenizer_rejects_non_string_input() -> None:
    with pytest.raises(ValueError, match="must be a string"):
        LocalTextTokenizer(fake_tokenizer()).tokenize(42)  # type: ignore[arg-type]


def test_local_tokenizer_requires_provisioned_asset(tmp_path: Path) -> None:
    with pytest.raises(TokenizerLoadError, match="tokenizer-setup"):
        load_local_tokenizer(root=tmp_path, asset=fake_asset(fake_tokenizer()))


def test_local_tokenizer_refuses_tampered_asset(tmp_path: Path) -> None:
    asset = fake_asset(fake_tokenizer())
    tokenizer_path(tmp_path, asset).write_text("tampered")
    with pytest.raises(TokenizerLoadError, match="checksum"):
        load_local_tokenizer(root=tmp_path, asset=asset)


def test_local_tokenizer_maps_invalid_asset_to_safe_error(tmp_path: Path) -> None:
    invalid = b"not valid tokenizer json"
    asset = TokenizerAsset(
        model="broken-model",
        revision="test-revision",
        sha256=hashlib.sha256(invalid).hexdigest(),
    )
    tokenizer_path(tmp_path, asset).write_bytes(invalid)
    with pytest.raises(TokenizerLoadError, match="could not be loaded"):
        load_local_tokenizer(root=tmp_path, asset=asset)


def test_loads_configured_tokenizer_without_api_key(tmp_path: Path, monkeypatch) -> None:
    tokenizer = fake_tokenizer()
    asset = fake_asset(tokenizer, COHERE_EMBED_ENGLISH_LIGHT_V3.model)
    monkeypatch.setitem(SUPPORTED_ASSETS, asset.model, asset)
    tokenizer_path(tmp_path, asset).write_text(tokenizer.to_str(pretty=True))

    local = load_configured_tokenizer(Settings(tokenizer_root=tmp_path, cohere_api_key=None))

    assert local.tokenize("remote work")[0].token_id == 1


def test_configured_load_rejects_unsupported_model(tmp_path: Path) -> None:
    settings = Settings(tokenizer_root=tmp_path, cohere_embed_model="unsupported")
    with pytest.raises(TokenizerLoadError, match="No pinned tokenizer"):
        load_configured_tokenizer(settings)


def test_local_tokenizer_rejects_unusable_offsets() -> None:
    broken = Mock()
    broken.encode.return_value = Mock(ids=[1], offsets=[(0, 0)])
    with pytest.raises(TokenizerLoadError, match="unusable character offset"):
        LocalTextTokenizer(broken).tokenize("remote")
