import pandas as pd
from pathlib import Path
from data_pipeline.config import PipelineConfig


class DataExporter:

    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()

    def save_checkpoint(self, movies: list, path: str = None) -> None:
        if not movies:
            return
        output = path or self.config.tmdb.output_path
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(movies).to_csv(output, index=False, encoding="utf-8")
        print(f"  Checkpoint — {len(movies):,} movies saved to {output}")

    def language_summary(self, df: pd.DataFrame) -> None:
        if "original_language" not in df.columns:
            return
        print("\nLanguage breakdown:")
        counts = df["original_language"].value_counts().head(10)
        for lang, count in counts.items():
            print(f"  {lang or 'unknown':6s} → {count:,}")

    def finalize(self, movies: list) -> pd.DataFrame:
        """
        Convert movies list to DataFrame, clean, and save.
        """
        if not movies:
            print("No movies to save.")
            return pd.DataFrame()

        df = pd.DataFrame(movies)

        before = len(df)
        df = df.dropna(subset=["title", "plotsummary"])
        df = df[df["plotsummary"].str.len() >= self.config.tmdb.min_overview_length]
        df = df.drop_duplicates(subset=["movie_id"])
        df = df.reset_index(drop=True)

        print(f"\nCleaning: {before:,} -> {len(df):,} movies")
        self.language_summary(df)

        output = self.config.final_output
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output, index=False, encoding="utf-8")

        print(f"\nFinal dataset saved to {output}")
        print(f"Total movies: {len(df):,}")
