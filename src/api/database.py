import logging
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from src.api.config import settings

logger = logging.getLogger(__name__)

# Base class for SQLAlchemy ORM models
Base = declarative_base()

# Configure Engine
db_url = settings.get_database_url()
is_sqlite = db_url.startswith("sqlite")

connect_args = {"check_same_thread": False} if is_sqlite else {}

pool_kwargs = {}
if not is_sqlite:
    pool_kwargs = {
        "pool_size": 20,
        "max_overflow": 10,
        "pool_recycle": 1800,
        "pool_timeout": 30,
    }

try:
    engine = create_engine(
        db_url,
        pool_pre_ping=True,
        echo=False,
        connect_args=connect_args,
        **pool_kwargs,
    )
except Exception as e:
    logger.warning(f"Failed to initialize database engine for URL: {db_url}. Error: {e}")
    # Fallback to in-memory sqlite if dialect is missing or configuration is incomplete
    engine = create_engine("sqlite:///:memory:", echo=False)

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
