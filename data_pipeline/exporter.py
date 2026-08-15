import pandas as pd
from pathlib import Path
from data_pipeline.config import pipeline_config


class DataExporter:

    def __init__(self, config=None):
        self.config = config or pipeline_config

    def save_checkpoint(self, movies: list, path: str = None) -> None:
        
        if not movies:
            return

        output = path or self.config.tmdb.output_path
        Path(output).parent.mkdir(parents=True, exist_ok=True)

        new_df = pd.DataFrame(movies)

        if Path(output).exists():
            try:
                existing = pd.read_csv(output, encoding="utf-8")
                combined = pd.concat([existing, new_df], ignore_index=True)
                combined = combined.drop_duplicates(subset=["movie_id"], keep="first")
            except Exception:
                # If existing file is corrupted/empty, just use new data
                combined = new_df
        else:
            combined = new_df

        combined.to_csv(output, index=False, encoding="utf-8")
        print(f"  Checkpoint — {len(combined):,} total movies saved to {output}")

    def language_summary(self, df: pd.DataFrame) -> None:
        if "original_language" not in df.columns:
            return
        print("\nLanguage breakdown:")
        counts = df["original_language"].value_counts().head(10)
        for lang, count in counts.items():
            print(f"  {lang or 'unknown':6s} -> {count:,}")

    def finalize(self, movies: list) -> pd.DataFrame:

        output = self.config.final_output

        if not movies:
            print("No new movies fetched.")
            if Path(output).exists():
                existing = pd.read_csv(output, encoding="utf-8")
                print(f"Existing file unchanged: {len(existing):,} movies")
                return existing
            return pd.DataFrame()

        new_df = pd.DataFrame(movies)

        # Clean the newly fetched batch
        before = len(new_df)
        new_df = new_df.dropna(subset=["title", "plotsummary"])
        new_df = new_df[new_df["plotsummary"].str.len() >= self.config.tmdb.min_overview_length]
        new_df = new_df.drop_duplicates(subset=["movie_id"])
        print(f"\nNewly fetched (after cleaning): {before:,} -> {len(new_df):,} movies")

        # Merge with existing file if present
        if Path(output).exists():
            existing = pd.read_csv(output, encoding="utf-8")
            print(f"Existing movies in {output}: {len(existing):,}")

            before_merge = len(existing) + len(new_df)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["movie_id"], keep="first")
            after_merge = len(combined)

            duplicates_skipped = before_merge - after_merge
            print(f"Duplicates skipped (already existed): {duplicates_skipped:,}")
        else:
            combined = new_df
            print(f"No existing file — creating {output} fresh")

        combined = combined.reset_index(drop=True)
        self.language_summary(combined)

        Path(output).parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output, index=False, encoding="utf-8")

        print(f"\nSaved to {output}")
        print(f"Total movies now: {len(combined):,}")

        return combined
