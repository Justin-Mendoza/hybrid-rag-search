from collections.abc import Coroutine
from typing import Any

import pytest

from hybrid_rag_search import seed


def test_seed_statements_cover_each_demo_record() -> None:
    statements = seed.seed_statements()

    assert [statement.table.name for statement in statements] == [
        "tenants",
        "users",
        "memberships",
        "collections",
        "collection_grants",
    ]


@pytest.mark.anyio
async def test_seed_database_executes_all_statements_and_disposes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executed: list[object] = []

    class Connection:
        async def execute(self, statement: object) -> None:
            executed.append(statement)

    class Transaction:
        async def __aenter__(self) -> Connection:
            return Connection()

        async def __aexit__(self, *_args: object) -> None:
            return None

    class Engine:
        disposed = False

        def begin(self) -> Transaction:
            return Transaction()

        async def dispose(self) -> None:
            self.disposed = True

    engine = Engine()
    statements = seed.seed_statements()
    monkeypatch.setattr(seed, "create_async_engine", lambda _url: engine)
    monkeypatch.setattr(seed, "seed_statements", lambda: statements)

    await seed.seed_database("postgresql://user:pass@localhost/app")

    assert executed == statements
    assert engine.disposed is True


def test_seed_main_runs_and_reports_success(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def close_coroutine(coroutine: Coroutine[Any, Any, None]) -> None:
        coroutine.close()

    monkeypatch.setattr(seed.asyncio, "run", close_coroutine)

    seed.main()

    assert "Seeded Acme Demo" in capsys.readouterr().out
