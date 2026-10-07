import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from requests.exceptions import ReadTimeout

from app.history import store_research
from app.research_graph import research
from app.settings import DEFAULT_PDF_DIR
from app.vector_store import ingest_pdfs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(title="GameVerse India Research API", version="1.0.0")


class ResearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "GameVerse India Research API"}


@app.post("/research")
async def run_research(request: ResearchRequest) -> dict[str, object]:
    try:
        result = await research(request.question)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ReadTimeout as exc:
        logger.warning("NVIDIA request timed out during research.", exc_info=True)
        raise HTTPException(
            status_code=504,
            detail=(
                "NVIDIA did not respond before the request timeout. Try again shortly, "
                "or increase NVIDIA_REQUEST_TIMEOUT_SECONDS in .env."
            ),
        ) from exc
    except Exception as exc:
        logger.exception("Research request failed.")
        raise HTTPException(
            status_code=502,
            detail="Research failed. Check the configured NVIDIA, Parallel, and database services.",
        ) from exc
    try:
        store_research(result)
    except Exception:
        logger.exception("Research completed, but PostgreSQL history could not be saved.")
        result["persistence_warning"] = (
            "Research completed, but it could not be saved to PostgreSQL. "
            "Check DATABASE_URL and the database connection."
        )
    return result


@app.post("/ingest")
def ingest_documents() -> dict[str, int]:
    try:
        return ingest_pdfs(DEFAULT_PDF_DIR)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("PDF ingestion failed.")
        raise HTTPException(status_code=500, detail="PDF ingestion failed; check server logs.") from exc
