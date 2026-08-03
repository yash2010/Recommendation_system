from data_pipeline.config import TMDBConfig

class TMDBProcessor:

    def __init__(self, config: TMDBConfig = None):
        self.config = config or TMDBConfig()

    def extract_director(self, credits: dict) -> str:
        if not credits:
            return "Unknown"
        for person in credits.get("crew", []):
            if person.get("job") == "Director":
                return person.get("name", "Unknown")
        return "Unknown"

    def extract_cast(self, credits: dict) -> str:
        if not credits:
            return ""
        cast = credits.get("cast", [])[:self.config.max_cast]
        return ", ".join(p.get("name", "") for p in cast if p.get("name"))

    def extract_genres(self, movie: dict) -> str:
        genres = movie.get("genres", [])
        if not genres:
            return "unknown"
        return "|".join(g["name"].lower() for g in genres if g.get("name"))

    def extract_poster_url(self, movie: dict) -> str:
        path = movie.get("poster_path", "")
        if not path or not self.config.include_posters:
            return ""
        return f"{self.config.poster_base_url}{path}"

    def extract_overview(self, raw: dict) -> str:
   
        overview = raw.get("overview", "").strip()
        if overview and len(overview) >= self.config.min_overview_length:
            return overview

        translations = raw.get("translations", {}).get("translations", [])
        for t in translations:
            if t.get("iso_639_1") == "en":
                translated = t.get("data", {}).get("overview", "").strip()
                if translated and len(translated) >= self.config.min_overview_length:
                    return translated

        # Fall back to any available translation
        for t in translations:
            translated = t.get("data", {}).get("overview", "").strip()
            if translated and len(translated) >= self.config.min_overview_length:
                return translated

        return overview

    def extract_year(self, movie: dict) -> int:
        date = movie.get("release_date", "")
        if date and len(date) >= 4:
            try:
                return int(date[:4])
            except ValueError:
                pass
        return 0

    def process(self, raw: dict) -> dict | None:
   
        if not raw:
            return None

        title    = raw.get("title", "").strip()
        overview = self.extract_overview(raw)

        if not title or not overview:
            return None
        if len(overview) < self.config.min_overview_length:
            return None

        credits  = raw.get("credits", {})

        return {
            "movie_id":          raw.get("id"),
            "title":             title,
            "release_year":      self.extract_year(raw),
            "genre":             self.extract_genres(raw),
            "director":          self.extract_director(credits),
            "cast":              self.extract_cast(credits),
            "plotsummary":       overview,
            "tmdb_rating":       round(raw.get("vote_average", 0.0), 1),
            "tmdb_votes":        raw.get("vote_count", 0),
            "poster_url":        self.extract_poster_url(raw),
            "original_language": raw.get("original_language", ""),
            "source":            "tmdb",
        }
