import runpy
from unittest.mock import MagicMock

from hybrid_rag_search.ingestion.recovery import RecoveryRunResult


def test_recovery_cli_reports_summary(monkeypatch, capsys) -> None:
    from hybrid_rag_search import ingestion_recovery

    monkeypatch.setattr(ingestion_recovery, "get_settings", MagicMock())
    monkeypatch.setattr(
        ingestion_recovery,
        "run_configured_recovery",
        MagicMock(),
    )
    monkeypatch.setattr(
        ingestion_recovery.asyncio,
        "run",
        MagicMock(return_value=RecoveryRunResult(2, 1, 4)),
    )
    ingestion_recovery.main()
    assert capsys.readouterr().out == ("ingestion recovery: requeued=2 failed=1 published=4\n")


def test_recovery_module_entry_point(monkeypatch, capsys) -> None:
    import asyncio
    import sys

    def run(coroutine):
        coroutine.close()
        return RecoveryRunResult(0, 0, 0)

    monkeypatch.setattr(
        asyncio,
        "run",
        run,
    )
    sys.modules.pop("hybrid_rag_search.ingestion_recovery", None)
    runpy.run_module("hybrid_rag_search.ingestion_recovery", run_name="__main__")
    assert capsys.readouterr().out.endswith("requeued=0 failed=0 published=0\n")
