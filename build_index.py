import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

CSV_PATH = "data/movies_clean.csv"
ARTIFACTS = Path("artifacts")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 128

print("Loading data...")
df = pd.read_csv(CSV_PATH, encoding="utf-8")

df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("/", "_")

df["cast"] = df["cast"].fillna("")
df["plotsummary"] = df["plotsummary"].fillna("")
df["genre"] = df["genre"].fillna("")
df["director"] = df["director"].fillna("")

def build_text(row) -> str:
    parts = []
    if row["title"]: parts.append(f"Title: {row['title']}")
    if row["cast"]: parts.append(f"Cast: {row['cast']}")
    if row["plotsummary"]: parts.append(f"Plot Summary: {row['plotsummary']}")
    if row["genre"]: parts.append(f"Genre: {row['genre']}")
    if row["director"]: parts.append(f"Director: {row['director']}")
    return " ".join(parts)

print("Building text...")
corpus = df.apply(build_text, axis=1).tolist()

print("\n Sample text:")
for i, text in enumerate(corpus[:5]):
    print(f"{i+1}. {text}")

print(f"\nLoading model... {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

print(f"Embedding {len(corpus)} movies...")
embeddings = model.encode(corpus, batch_size=BATCH_SIZE, show_progress_bar=True, normalize_embeddings=True, convert_to_numpy=True)
embeddings = embeddings.astype(np.float32)

print(f"\nEmbeddings shape: {embeddings.shape}")

ARTIFACTS.mkdir(exist_ok=True)
np.save(ARTIFACTS / "embeddings.npy", embeddings)
(ARTIFACTS / "model_name.txt").write_text(MODEL_NAME)
print("Embeddings saved.")
df.to_parquet(ARTIFACTS / "movies.parquet", index=False)

print(f"\nSaved to {ARTIFACTS}/")
print("  embeddings.npy  →", embeddings.nbytes // 1_000_000, "MB")
print("  movies.parquet")
print("  model_name.txt")
print("\nDone. Run recommender.py next.")
