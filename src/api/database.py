import logging
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from src.api.config import settings

logger = logging.getLogger(__name__)

# Base class for SQLAlchemy ORM models
Base = declarative_base()

def init_database_engine():
    db_url = settings.get_database_url()
    is_sqlite = db_url.startswith("sqlite")
    connect_args = {"check_same_thread": False} if is_sqlite else {}
    pool_kwargs = {}
    if not is_sqlite:
        pool_kwargs = {
            "pool_size": 20,
            "max_overflow": 10,
            "pool_recycle": 1800,
            "pool_timeout": 3,
        }

    try:
        eng = create_engine(
            db_url,
            pool_pre_ping=True,
            echo=False,
            connect_args=connect_args,
            **pool_kwargs,
        )
        if not is_sqlite:
            # Probe connection to verify PostgreSQL is running
            with eng.connect() as conn:
                pass
        return eng
    except Exception as e:
        logger.warning(
            f"Could not connect to configured database ({db_url}): {e}. "
            "Falling back to local SQLite database (agri_yield_local.db)."
        )
        sqlite_engine = create_engine("sqlite:///./agri_yield_local.db", connect_args={"check_same_thread": False}, echo=False)
        try:
            import src.api.models  # noqa: F401
            Base.metadata.create_all(bind=sqlite_engine)
        except Exception:
            pass
        return sqlite_engine


engine = init_database_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, expire_on_commit=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency yielding a SQLAlchemy DB session per request,
    ensuring cleanup and teardown.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
