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

    def merge_with_wikipedia(self, tmdb_df: pd.DataFrame) -> pd.DataFrame:
        if not self.config.wiki.enabled:
            return tmdb_df

        wiki_path = self.config.wiki.wiki_path
        if not Path(wiki_path).exists():
            print(f"Wikipedia file not found at {wiki_path} — skipping merge")
            return tmdb_df

        print("\nMerging with Wikipedia fallback...")
        wiki_df = pd.read_csv(wiki_path, encoding="utf-8")

        wiki_df.columns = (
            wiki_df.columns.str.strip().str.lower()
            .str.replace(" ", "_").str.replace("/", "_")
        )

        for col, default in [
            ("tmdb_rating",        0.0),
            ("tmdb_votes",         0),
            ("poster_url",         ""),
            ("original_language",  ""),
        ]:
            if col not in wiki_df.columns:
                wiki_df[col] = default

        wiki_df["source"] = "wikipedia"

        tmdb_titles  = set(tmdb_df["title"].str.lower().str.strip())
        missing_mask = ~wiki_df["title"].str.lower().str.strip().isin(tmdb_titles)
        missing_wiki = wiki_df[missing_mask].copy()

        print(f"  TMDB movies:             {len(tmdb_df):,}")
        print(f"  Wikipedia-only movies:   {len(missing_wiki):,}")

        combined = pd.concat([tmdb_df, missing_wiki], ignore_index=True)
        combined = combined.drop_duplicates(subset=["title", "release_year"], keep="first")
        combined = combined.reset_index(drop=True)

        print(f"  Final combined total:    {len(combined):,}")
        return combined

    def language_summary(self, df: pd.DataFrame) -> None:
        if "original_language" not in df.columns:
            return
        print("\nLanguage breakdown:")
        counts = df["original_language"].value_counts().head(10)
        for lang, count in counts.items():
            print(f"  {lang or 'unknown':6s} → {count:,}")

    def finalize(self, movies: list) -> pd.DataFrame:
        if not movies:
            print("No movies to save.")
            return pd.DataFrame()

        df = pd.DataFrame(movies)

        before = len(df)
        df = df.dropna(subset=["title", "plotsummary"])
        df = df[df["plotsummary"].str.len() >= self.config.tmdb.min_overview_length]
        df = df.drop_duplicates(subset=["movie_id"])
        df = df.reset_index(drop=True)

        print(f"\nCleaning: {before:,} → {len(df):,} movies")
        self.language_summary(df)

        df = self.merge_with_wikipedia(df)

        output = self.config.final_output
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output, index=False, encoding="utf-8")

        print(f"\nFinal dataset saved to {output}")
        print(f"Total movies: {len(df):,}")

        return df
