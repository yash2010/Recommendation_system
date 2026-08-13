

import time
import pandas as pd
from pathlib import Path
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

    def _load_existing_ids(self) -> set:

        output = self.config.final_output
        if not Path(output).exists():
            return set()

        try:
            existing = pd.read_csv(output, usecols=["movie_id"], encoding="utf-8")
            ids = set(existing["movie_id"].dropna().astype(int))
            print(f"Loaded {len(ids):,} existing movie_ids — these will be skipped\n")
            return ids
        except Exception as e:
            print(f"Could not read existing file for ID skip check: {e}")
            return set()

    def _fetch_language(
        self,
        language:     str,
        seen_ids:     set,
        existing_ids: set,
        movies:       list,
    ) -> None:
       
        cfg       = self.config.tmdb
        lang_name = LANGUAGE_NAMES.get(language, language)

        print(f"\n {lang_name} ({language})")
        print(f"   min_votes={cfg.min_votes}, max_pages={cfg.max_pages}")

        new_for_language     = 0
        skipped_already_have = 0

        for page in range(1, cfg.max_pages + 1):
            raw_results = self.fetcher.discover_page(page=page, language=language)

            if not raw_results:
                print(f"   Page {page}: no results — stopping {lang_name}")
                break

            for movie in raw_results:
                movie_id = movie.get("id")
                if not movie_id:
                    continue

                # Skip if already fetched in a previous run
                if movie_id in existing_ids:
                    skipped_already_have += 1
                    continue

                # Skip if already fetched earlier in THIS run
                # (e.g. same movie appears in both en and ta discover pages)
                if movie_id in seen_ids:
                    continue
                seen_ids.add(movie_id)

                # Only reaches here for genuinely new movies —
                # this is the expensive call we're trying to minimize
                raw = self.fetcher.movie_details(movie_id)
                time.sleep(cfg.delay)

                processed = self.processor.process(raw)
                if processed:
                    movies.append(processed)
                    new_for_language += 1

            if page % cfg.log_every == 0:
                print(f"   Page {page}/{cfg.max_pages} — "
                      f"{new_for_language} new, {skipped_already_have} skipped (already have)")

            if len(movies) > 0 and len(movies) % cfg.checkpoint_every < 20:
                self.exporter.save_checkpoint(movies, cfg.output_path)

        print(f"   {lang_name} complete: {new_for_language} new movies, "
              f"{skipped_already_have} skipped (already had them)")

    def run_discover(self) -> list:
       
        movies       = []
        seen_ids     = set()
        cfg          = self.config.tmdb
        existing_ids = self._load_existing_ids()

        estimated = len(cfg.languages) * cfg.max_pages * 20 * cfg.delay / 3600
        print(f"Languages: {cfg.languages}")
        print(f"Estimated time (upper bound, before ID-skip savings): ~{estimated:.1f} hours\n")

        for language in cfg.languages:
            self._fetch_language(language, seen_ids, existing_ids, movies)

        return movies

    def run_from_exports(self) -> list:

        cfg          = self.config.tmdb
        movies       = []
        seen_ids     = set()
        existing_ids = self._load_existing_ids()

        print("Phase 1: Downloading TMDB export file...")
        all_ids = self.fetcher.export_ids()

        if not all_ids:
            print("Export failed — falling back to discover mode")
            return self.run_discover()

        ids_to_fetch = [mid for mid in all_ids if mid not in existing_ids]
        skipped      = len(all_ids) - len(ids_to_fetch)
        print(f"\nExport IDs: {len(all_ids):,} total, "
              f"{skipped:,} already have, {len(ids_to_fetch):,} to fetch")

        for i, movie_id in enumerate(ids_to_fetch):
            if movie_id in seen_ids:
                continue
            seen_ids.add(movie_id)

            raw       = self.fetcher.movie_details(movie_id)
            time.sleep(cfg.delay)

            processed = self.processor.process(raw)
            if processed:
                movies.append(processed)

            if (i + 1) % 500 == 0:
                print(f"  {i+1:,}/{len(ids_to_fetch):,} — {len(movies):,} valid new movies")
                self.exporter.save_checkpoint(movies, cfg.output_path)

        print(f"\nExport phase complete: {len(movies):,} new movies")

        regional = [l for l in cfg.languages if l != "en"]
        if regional:
            print(f"\nPhase 2: Fetching regional languages: {regional}")
            for language in regional:
                self._fetch_language(language, seen_ids, existing_ids, movies)

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

        print(f"\nFetch complete — {len(movies):,} NEW movies collected")

        self.exporter.finalize(movies)
