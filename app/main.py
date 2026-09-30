import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.recommender import MovieRecommender

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "movies.csv")
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = FastAPI(title="Movie Recommender API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

recommender = MovieRecommender(CSV_PATH)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/movies")
def get_movies():
    """List all movies available in the catalog."""
    return {"movies": recommender.list_titles()}


@app.get("/api/recommend")
def recommend(
    title: str = Query(..., description="Movie title to base recommendations on"),
    top_n: int = Query(5, ge=1, le=20, description="Number of recommendations"),
):
    """Return top_n movies most similar to the given title."""
    results = recommender.get_recommendations(title, top_n=top_n)
    if results is None:
        raise HTTPException(
            status_code=404,
            detail=f"Movie '{title}' not found. Try GET /api/movies to see available titles.",
        )
    return {"query": title, "recommendations": results}
