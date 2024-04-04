from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from pirate_speak.chain import chain as pirate_speak_chain

app = FastAPI()


@app.get("/")
async def redirect_root_to_docs() -> RedirectResponse:
    return RedirectResponse("/docs")


# Add routes to other langchain templates here
add_routes(app, pirate_speak_chain, path="/pirate-speak")
