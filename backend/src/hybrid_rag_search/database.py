from sqlalchemy import MetaData
from sqlalchemy.engine import URL, make_url
from sqlalchemy.orm import DeclarativeBase

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Declarative base shared by application models and Alembic."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


def async_database_url(database_url: str) -> URL:
    """Return a SQLAlchemy URL using the project's async PostgreSQL driver."""
    url = make_url(database_url)
    if url.drivername == "postgresql":
        return url.set(drivername="postgresql+asyncpg")
    return url
