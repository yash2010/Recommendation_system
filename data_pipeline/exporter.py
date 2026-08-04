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
        if not movies:
            print("No new movies to save.")
            return pd.DataFrame()

        df = pd.DataFrame(movies)
        df = df.dropna(subset=["title", "plotsummary"])
        df = df[df["plotsummary"].str.len() >= self.config.tmdb.min_overview_length]

        output = self.config.final_output

        if Path(output).exists():
            existing = pd.read_csv(output, encoding="utf-8")
            print(f"Existing movies: {len(existing):,}")
            df = pd.concat([existing, df], ignore_index=True)

        df = df.drop_duplicates(subset=["movie_id"], keep="first")
        df = df.reset_index(drop=True)

        Path(output).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output, index=False, encoding="utf-8")

        print(f"Total movies after merge: {len(df):,}")
        return df