"""FastAPI server for langchain templates."""

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from pirate_speak.chain import chain as pirate_speak_chain
from rag_chroma_multi_modal_multi_vector import (
    chain as rag_chroma_multi_modal_multi_vector_chain,
)

app = FastAPI()


@app.get("/")
async def redirect_root_to_docs() -> RedirectResponse:
    """Redirect root to /docs endpoint."""
    return RedirectResponse("/docs")


# Add routes to other langchain templates here
add_routes(app, pirate_speak_chain, path="/pirate-speak")

add_routes(
    app,
    rag_chroma_multi_modal_multi_vector_chain,
    path="/rag-chroma-multi-modal-multi-vector",
)
