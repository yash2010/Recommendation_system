"""
fetch_tmdb.py

Entry point for the TMDB data pipeline.

Usage:
    python fetch_tmdb.py --pages 10                     # quick test
    python fetch_tmdb.py --pages 500                    # ~10k movies
    python fetch_tmdb.py --use-exports                  # ~80k overnight
    python fetch_tmdb.py --languages en,ta,hi,te,ml     # more languages
    python fetch_tmdb.py --min-votes 20                 # stricter filter
"""

import argparse
from data_pipeline.pipeline import TMDBPipeline
from data_pipeline.config import PipelineConfig, TMDBConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Fetch movies from TMDB")

    parser.add_argument("--pages",       type=int,  default=500)
    parser.add_argument("--languages",   type=str,  default="en,ta")
    parser.add_argument("--min-votes",   type=int,  default=10)
    parser.add_argument("--use-exports", action="store_true")
    parser.add_argument("--output",      type=str,  default="data/tmdb_movies.csv")
    parser.add_argument("--final-output",type=str,  default="data/movies_final.csv")

    return parser.parse_args()


def main():
    args      = parse_args()
    languages = [l.strip() for l in args.languages.split(",") if l.strip()]

    config = PipelineConfig(
        tmdb = TMDBConfig(
            languages    = languages,
            use_exports  = args.use_exports,
            max_pages    = args.pages,
            min_votes    = args.min_votes,
            output_path  = args.output,
        ),
        final_output = args.final_output,
    )

    TMDBPipeline(config).run()


if __name__ == "__main__":
    main()