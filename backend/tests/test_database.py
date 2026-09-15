from hybrid_rag_search.database import NAMING_CONVENTION, Base, async_database_url
from hybrid_rag_search.models import (
    AuditEvent,
    Collection,
    CollectionGrant,
    CollectionPermission,
    Document,
    DocumentStatus,
    EvaluationDataset,
    EvaluationRun,
    EvaluationRunStatus,
    IngestionJob,
    IngestionJobStatus,
    Judgment,
    JudgmentTarget,
    Membership,
    MembershipRole,
    QueryTrace,
    RetrievalConfig,
    Tenant,
    User,
)


def test_base_uses_stable_constraint_naming_convention() -> None:
    assert Base.metadata.naming_convention == NAMING_CONVENTION
    assert NAMING_CONVENTION["pk"] == "pk_%(table_name)s"
    assert NAMING_CONVENTION["fk"].startswith("fk_%(table_name)s")


def test_async_database_url_selects_asyncpg() -> None:
    assert (
        async_database_url("postgresql://user:pass@localhost/app").drivername
        == "postgresql+asyncpg"
    )
    assert (
        async_database_url("postgresql+asyncpg://user:pass@localhost/app").drivername
        == "postgresql+asyncpg"
    )


def test_identity_models_register_expected_tables_and_roles() -> None:
    assert {Tenant.__tablename__, User.__tablename__, Membership.__tablename__} <= set(
        Base.metadata.tables
    )
    assert {role.value for role in MembershipRole} == {"owner", "admin", "member"}


def test_content_models_register_expected_tables_and_states() -> None:
    assert {
        Collection.__tablename__,
        CollectionGrant.__tablename__,
        Document.__tablename__,
        IngestionJob.__tablename__,
    } <= set(Base.metadata.tables)
    assert {permission.value for permission in CollectionPermission} == {
        "read",
        "write",
        "manage",
    }
    assert {status.value for status in DocumentStatus} == {
        "pending",
        "processing",
        "ready",
        "failed",
        "deleting",
        "deleted",
    }
    assert {status.value for status in IngestionJobStatus} == {
        "queued",
        "running",
        "succeeded",
        "failed",
        "cancelled",
    }


def test_retrieval_models_register_expected_tables() -> None:
    assert {RetrievalConfig.__tablename__, QueryTrace.__tablename__} <= set(Base.metadata.tables)


def test_evaluation_models_register_expected_tables_and_states() -> None:
    assert {
        EvaluationDataset.__tablename__,
        Judgment.__tablename__,
        EvaluationRun.__tablename__,
        AuditEvent.__tablename__,
    } <= set(Base.metadata.tables)
    assert {target.value for target in JudgmentTarget} == {"document", "chunk"}
    assert {status.value for status in EvaluationRunStatus} == {
        "queued",
        "running",
        "succeeded",
        "failed",
        "cancelled",
    }
