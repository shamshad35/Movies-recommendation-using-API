"""
Content-based movie recommender.

Builds a "content soup" for each movie (genres + director + cast + overview),
vectorizes it with TF-IDF, and recommends movies with the highest
cosine similarity to a given title.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MovieRecommender:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self.df["content"] = self._build_content_soup(self.df)

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["content"])
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)

        # Map lowercased title -> row index, for fast lookup
        self.title_to_index = {
            title.lower(): idx for idx, title in enumerate(self.df["title"])
        }

    @staticmethod
    def _build_content_soup(df: pd.DataFrame) -> pd.Series:
        """Combine relevant text fields into one string per movie."""
        def row_to_soup(row):
            genres = str(row.get("genres", "")).replace(",", " ")
            director = str(row.get("director", "")).replace(" ", "")
            cast = " ".join(
                name.strip().replace(" ", "")
                for name in str(row.get("cast", "")).split(",")
            )
            overview = str(row.get("overview", ""))
            # Repeat genres/director to give them more weight than the overview
            return f"{genres} {genres} {director} {director} {cast} {overview}"

        return df.apply(row_to_soup, axis=1)

    def list_titles(self):
        return self.df[["id", "title", "year", "genres"]].to_dict(orient="records")

    def get_recommendations(self, title: str, top_n: int = 5):
        key = title.lower().strip()
        if key not in self.title_to_index:
            return None  # signal "not found" to the caller

        idx = self.title_to_index[key]
        scores = list(enumerate(self.similarity_matrix[idx]))
        # Exclude the movie itself, sort by similarity descending
        scores = [s for s in scores if s[0] != idx]
        scores.sort(key=lambda x: x[1], reverse=True)
        top_matches = scores[:top_n]

        results = []
        for movie_idx, score in top_matches:
            row = self.df.iloc[movie_idx]
            results.append({
                "id": int(row["id"]),
                "title": row["title"],
                "year": int(row["year"]),
                "genres": row["genres"],
                "similarity": round(float(score), 4),
            })
        return results
