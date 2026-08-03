"""
data_pipeline/pipeline.py
"""

import time
from data_pipeline.config import PipelineConfig, LANGUAGE_NAMES
from data_pipeline.fetcher import TMDBFetcher
from data_pipeline.processor import TMDBProcessor
from data_pipeline.exporter import DataExporter


class TMDBPipeline:

    def __init__(self, config: PipelineConfig = None):
        self.config    = config or PipelineConfig()
        self.fetcher   = TMDBFetcher(self.config.tmdb)
        self.processor = TMDBProcessor(self.config.tmdb)
        self.exporter  = DataExporter(self.config)

    def _fetch_language(self, language: str, seen_ids: set, movies: list) -> None:
        cfg       = self.config.tmdb
        lang_name = LANGUAGE_NAMES.get(language, language)

        print(f"\n {lang_name} ({language})")
        print(f"   min_votes={cfg.min_votes}, max_pages={cfg.max_pages}")

        new_for_language = 0

        for page in range(1, cfg.max_pages + 1):
            raw_results = self.fetcher.discover_page(page=page, language=language)

            if not raw_results:
                print(f"   Page {page}: no results — stopping {lang_name}")
                break

            for movie in raw_results:
                movie_id = movie.get("id")
                if not movie_id or movie_id in seen_ids:
                    continue
                seen_ids.add(movie_id)

                raw       = self.fetcher.movie_details(movie_id)
                time.sleep(cfg.delay)

                processed = self.processor.process(raw)
                if processed:
                    movies.append(processed)
                    new_for_language += 1

            if page % cfg.log_every == 0:
                print(f"   Page {page}/{cfg.max_pages} — {new_for_language} {lang_name} movies so far")

            if len(movies) > 0 and len(movies) % cfg.checkpoint_every < 20:
                self.exporter.save_checkpoint(movies, cfg.output_path)

        print(f"   {lang_name} complete: {new_for_language} movies added")

    def run_discover(self) -> list:
        movies   = []
        seen_ids = set()
        cfg      = self.config.tmdb

        estimated = len(cfg.languages) * cfg.max_pages * 20 * cfg.delay / 3600
        print(f"Languages: {cfg.languages}")
        print(f"Estimated time: ~{estimated:.1f} hours\n")

        for language in cfg.languages:
            self._fetch_language(language, seen_ids, movies)

        return movies

    def run_from_exports(self) -> list:
        cfg      = self.config.tmdb
        movies   = []
        seen_ids = set()

        print("Phase 1: Downloading TMDB export file...")
        all_ids = self.fetcher.export_ids()

        if not all_ids:
            print("Export failed — falling back to discover mode")
            return self.run_discover()

        print(f"\nFetching details for {len(all_ids):,} movies...")

        for i, movie_id in enumerate(all_ids):
            if movie_id in seen_ids:
                continue
            seen_ids.add(movie_id)

            raw       = self.fetcher.movie_details(movie_id)
            time.sleep(cfg.delay)

            processed = self.processor.process(raw)
            if processed:
                movies.append(processed)

            if (i + 1) % 500 == 0:
                print(f"  {i+1:,}/{len(all_ids):,} — {len(movies):,} valid movies")
                self.exporter.save_checkpoint(movies, cfg.output_path)

        print(f"\nExport phase complete: {len(movies):,} movies")

        regional = [l for l in cfg.languages if l != "en"]
        if regional:
            print(f"\nPhase 2: Fetching regional languages: {regional}")
            for language in regional:
                self._fetch_language(language, seen_ids, movies)

        return movies

    def run(self) -> None:
        cfg = self.config.tmdb

        if not cfg.token:
            print("ERROR: TMDB_TOKEN not set.")
            print("Add this to your .env file:")
            print("  TMDB_TOKEN=your_token_here")
            return

        print("=" * 55)
        print("  TMDB Movie Data Pipeline")
        print("=" * 55)
        print(f"  Mode:      {'Export file' if cfg.use_exports else 'Discover API'}")
        print(f"  Languages: {cfg.languages}")
        print(f"  Output:    {self.config.final_output}")
        print("=" * 55)

        if cfg.use_exports:
            movies = self.run_from_exports()
        else:
            movies = self.run_discover()

        print(f"\nFetch complete — {len(movies):,} movies collected")

        self.exporter.finalize(movies)
