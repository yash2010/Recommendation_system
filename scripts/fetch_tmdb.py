import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_pipeline.pipeline import TMDBPipeline
from data_pipeline.config import tmdb_config, pipeline_config


def parse_args():
    parser = argparse.ArgumentParser(description="Fetch movies from TMDB")

    parser.add_argument("--pages",       type=int,  default=500)
    parser.add_argument("--languages",   type=str,  default="en,ta")
    parser.add_argument("--min-votes",   type=int,  default=10)
    parser.add_argument("--use-exports", action="store_true")
    parser.add_argument("--output",      type=str,  default="data/tmdb_movies.csv")

    return parser.parse_args()


def main():
    args      = parse_args()
    languages = [l.strip() for l in args.languages.split(",") if l.strip()]

    tmdb_config.languages   = languages
    tmdb_config.use_exports = args.use_exports
    tmdb_config.max_pages   = args.pages
    tmdb_config.min_votes   = args.min_votes
    tmdb_config.output_path = args.output

    pipeline_config.final_output = args.output

    TMDBPipeline().run()


if __name__ == "__main__":
    main()
