import runpy
from pathlib import Path
from unittest.mock import Mock

import pytest

from hybrid_rag_search import config, tokenization
from hybrid_rag_search.tokenization import TokenizerProvisionResult, TokenizerSetupError


def test_module_entry_point_exits_with_main_result(monkeypatch, capsys) -> None:
    monkeypatch.setattr(config, "get_settings", Mock())
    monkeypatch.setattr(
        tokenization,
        "provision_configured_tokenizer",
        Mock(side_effect=TokenizerSetupError("entry-point failure")),
    )
    with pytest.raises(SystemExit) as caught:
        runpy.run_module("hybrid_rag_search.tokenizer_setup", run_name="__main__")
    assert caught.value.code == 1
    assert capsys.readouterr().out == "Tokenizer setup failed: entry-point failure\n"


def test_cli_reports_download(monkeypatch, capsys, tmp_path: Path) -> None:
    from hybrid_rag_search.tokenizer_setup import main

    result = TokenizerProvisionResult(tmp_path / "tokenizer.json", "abc", downloaded=True)
    monkeypatch.setattr("hybrid_rag_search.tokenizer_setup.get_settings", Mock())
    monkeypatch.setattr(
        "hybrid_rag_search.tokenizer_setup.provision_configured_tokenizer",
        Mock(return_value=result),
    )
    assert main() == 0
    assert capsys.readouterr().out == (
        f"Tokenizer downloaded: {result.path} (sha256={result.sha256})\n"
    )


def test_cli_reports_verified_existing_asset(monkeypatch, capsys, tmp_path: Path) -> None:
    from hybrid_rag_search.tokenizer_setup import main

    result = TokenizerProvisionResult(tmp_path / "tokenizer.json", "abc", downloaded=False)
    monkeypatch.setattr("hybrid_rag_search.tokenizer_setup.get_settings", Mock())
    monkeypatch.setattr(
        "hybrid_rag_search.tokenizer_setup.provision_configured_tokenizer",
        Mock(return_value=result),
    )
    assert main() == 0
    assert "already verified" in capsys.readouterr().out


def test_cli_reports_safe_failure(monkeypatch, capsys) -> None:
    from hybrid_rag_search.tokenizer_setup import main

    monkeypatch.setattr("hybrid_rag_search.tokenizer_setup.get_settings", Mock())
    monkeypatch.setattr(
        "hybrid_rag_search.tokenizer_setup.provision_configured_tokenizer",
        Mock(side_effect=TokenizerSetupError("safe message")),
    )
    assert main() == 1
    assert capsys.readouterr().out == "Tokenizer setup failed: safe message\n"
