import os
import sys
sys.path.append("..")

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from backend.config import COLLECTION_NAME, STATIC_DIR
from backend.neural_searcher import NeuralSearcher


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

neural_searcher = NeuralSearcher(collection_name=COLLECTION_NAME)

@app.get("/api/search")
async def read_item(q: str):
    return {
        "result": neural_searcher.search(question=q)
    }

# Mount the static files directory once the search endpoint is defined
if os.path.exists(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True))

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)