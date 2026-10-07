import json
import logging

from sqlalchemy import create_engine, make_url, text

from app.settings import optional_setting

logger = logging.getLogger(__name__)


def store_research(result: dict[str, object]) -> None:
    database_url = optional_setting("DATABASE_URL")
    if not database_url:
        return
    url = make_url(database_url)
    if url.drivername in {"postgres", "postgresql"}:
        url = url.set(drivername="postgresql+psycopg")
    engine = create_engine(url, pool_pre_ping=True)
    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS research_history (
                        id BIGSERIAL PRIMARY KEY,
                        question TEXT NOT NULL,
                        answer TEXT NOT NULL,
                        citations JSONB NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
            )
            connection.execute(
                text(
                    """
                    INSERT INTO research_history (question, answer, citations)
                    VALUES (:question, :answer, CAST(:citations AS JSONB))
                    """
                ),
                {
                    "question": result["question"],
                    "answer": result["answer"],
                    "citations": json.dumps(result.get("citations", [])),
                },
            )
    except Exception:
        logger.exception("Could not save the research session to PostgreSQL.")
        raise
    finally:
        engine.dispose()
