""" HTTP API and a simple web page. RUn unicorn docchat.api: app -- reoload"""
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from .llm import MissingAPIKey

from .pipeline import DocChat
load_dotenv()  # Load environment variables from .env file
# This is the absolute path pointing to the static folder holding index.html
STATIC = Path(__file__).parent / "static"

# Pydantic data schemas

class AskRequest(BaseModel):
    # Enforces questions is at least 1 char
    question: str = Field(min_length=1, max_length=2000)

class SourceOut(BaseModel):
    source: str
    page: int
    text: str

class AskResponse(BaseModel):
    answer: str
    sources: list[SourceOut]

class IngestRequest(BaseModel):
    folder: str


# THE APP
def create_app(docchat: DocChat | None = None) -> FastAPI:
    """ App factory: tests pass in a Docchat with fakes, production pases nothing"""
    # Instantiate the fastapi we app with title
    app = FastAPI(title="DocChat")
    # mutable state dictionary allowing lazy initialization of docchat
    state = {"docchat": docchat}

    def get_docchat() -> DocChat:
        # if no instance was injected (production), instantiate the production engine
        if state["docchat"] is None:
            state["docchat"] = DocChat()
        return state["docchat"]
    # The endpoints
    @app.get("/")
    def index() -> FileResponse:
        # Serves the single-page HTML chat interface
        return FileResponse(STATIC/ "index.html")
    @app.get("/health")
    def health() -> dict:
        # Health check endpoint returning status and current total chunks in vector store
        return {"status": "ok", "chunks": get_docchat().store.count()} 

    @app.post("/ingest")
    def ingest(req: IngestRequest) -> dict:
        # convert path to object
        folder = Path(req.folder)
        if not folder.is_dir():
            raise HTTPException(status_code=400, detail=f"Not a folder: {req.folder}")
        return {"chunks": get_docchat().ingest_folder(folder)}

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest) -> AskResponse:
        try:
            # query the RAG pipeline with the validated question
            result = get_docchat().ask(req.question)
        except MissingAPIKey as exc:
            # 503 Service Unavailable: correctly informs caller that the server lacks API credentials
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return AskResponse(
            answer=result.text,
            sources=[SourceOut(source=c.source, page=c.page, text=c.text) for c in result.sources],
        )
    return app

# Production entry point: instantiated with default dependencies for Uvicorn
app = create_app()