import os
import sys
sys.path.append("..")

from fastapi import FastAPI, Query
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
async def read_item(
    q: str = Query(..., title="User Question", description="Question asked by the user from the frontend"),
    num_results: int = Query(..., title="Number of Results", description="Number of results to return")
):
    return {
        "result": neural_searcher.search(question=q, num_results=num_results)
    }

# Mount the static files directory once the search endpoint is defined
if os.path.exists(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True))

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)