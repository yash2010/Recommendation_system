import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

@dataclass
class dataConfig:

    token: str = os.environ.get("TMDB_API_TOKEN", "")
    base_url: str = "https://api.themoviedb.org/3"
    delay: int = 0.25
    timeout: int = 10
    max_retries: int = 3

    language: list = field(default_factory=lambda: ["en-US", "ko-KR", "ta-IN"])

    min_votes: int = 20
    min_popularity: float = 2.0

    sort_by: str = "popularity.desc"
    max_pages: int = 500

    use_exports: bool = False
    export_min_popularity: float = 5.0
    export_max_movies: int = 80000

    max_cast: int = 3
    min_overview_length: int = 30
    include_posters: bool = True
    poster_base_url: str = "https://image.tmdb.org/t/p/w500"

    output_path: str = "data/tmdb_movies.csv"
    checkpoint_every: int = 500
    log_every: int = 10
    
