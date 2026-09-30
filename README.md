# Movie Recommender (Content-Based)

A content-based movie recommendation system with a FastAPI backend and a
simple web frontend. Recommends movies similar to one you pick, based on
genres, director, cast, and plot overview (TF-IDF + cosine similarity).

## How it works

1. Each movie's genres, director, cast, and overview are combined into one
   text "soup" (genres/director are repeated to weight them more heavily).
2. `TfidfVectorizer` (scikit-learn) turns each soup into a vector.
3. `cosine_similarity` compares every movie against every other movie.
4. To recommend, we look up the target movie's row and return the
   highest-scoring other movies.

## Project structure

```
movie-recs/
├── app/
│   ├── main.py          # FastAPI app & routes
│   └── recommender.py   # TF-IDF + cosine similarity engine
├── data/
│   └── movies.csv        # sample dataset (30 movies) — swap for a bigger one
├── static/
│   └── index.html         # simple frontend (dropdown + recommend button)
├── requirements.txt
└── README.md
```

## Setup

```bash
cd movie-recs
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
uvicorn app.main:app --reload
```

Then open **http://localhost:8000** in your browser — pick a movie from the
dropdown and click "Recommend."

## API endpoints

- `GET /api/movies` — list all movies in the catalog
- `GET /api/recommend?title=Inception&top_n=5` — get top 5 similar movies

Example:
```bash
curl "http://localhost:8000/api/recommend?title=Inception&top_n=5"
```

## Using a real dataset (recommended next step)

The included `movies.csv` has 30 movies just to demonstrate the system.
For something production-worthy, swap it for a real dataset:

- **[MovieLens](https://grouplens.org/datasets/movielens/)** (25M ratings +
  metadata) — best free option to start with
- **[TMDB](https://www.themoviedb.org/documentation/api)** — richer metadata
  (cast, crew, keywords, posters) via free API

Just make sure your CSV has: `id, title, year, genres, director, cast, overview`
(or update `recommender.py`'s `_build_content_soup` to match your columns).

## Next steps to level this up

1. **Collaborative filtering** — once you have real user ratings, add a
   second model (e.g. via the `surprise` or `implicit` library) that
   recommends based on what similar *users* liked, not just similar movies.
2. **Hybrid** — blend content-based and collaborative scores for better
   results (this is what most real systems do).
3. **Posters/images** — pull poster URLs from TMDB and show them in the
   frontend.
4. **Persistence** — swap the CSV for a real database (Postgres) as your
   catalog grows.
5. **Deploy** — host the FastAPI app on Render/Railway/Fly.io, and the
   static frontend can be served from the same app (as it is now) or split
   out to Vercel/Netlify pointing at the API.
