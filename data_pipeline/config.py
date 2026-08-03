import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

LANGUAGE_NAMES = {
    "en": "English",
    "ta": "Tamil",
    "ko": "Korean",
}

@dataclass
class TMDBConfig:
    token:          str   = os.environ.get("TMDB_TOKEN", "")
    base_url:       str   = "https://api.themoviedb.org/3"
    delay:          float = 0.26
    timeout:        int   = 10
    max_retries:    int   = 3
    languages:      list  = field(default_factory=lambda: ["en", "ta", "ko"])
    min_votes:      int   = 10
    min_popularity: float = 2.0
    max_pages:      int   = 500
    sort_by:        str   = "popularity.desc"
    use_exports:            bool  = False
    export_min_popularity:  float = 5.0
    export_max_movies:      int   = 80000
    max_cast:               int   = 3
    min_overview_length:    int   = 30
    include_posters:        bool  = True
    poster_base_url:        str   = "https://image.tmdb.org/t/p/w500"
    output_path:            str   = "data/tmdb_movies.csv"
    checkpoint_every:       int   = 500
    log_every:              int   = 10


@dataclass
class WikipediaFallbackConfig:
    enabled:    bool  = True
    wiki_path:  str   = "data/movies_clean.csv"


@dataclass
class PipelineConfig:
    tmdb:         TMDBConfig              = field(default_factory=TMDBConfig)
    wiki:         WikipediaFallbackConfig = field(default_factory=WikipediaFallbackConfig)
    final_output: str                     = "data/movies_final.csv"


tmdb_config     = TMDBConfig()
wiki_config     = WikipediaFallbackConfig()
pipeline_config = PipelineConfig()