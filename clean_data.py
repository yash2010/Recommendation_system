import pandas as pd

df = pd.read_csv("data/movies_with_plots.csv", encoding="utf-8")

df.columns = (df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("/", "_"))
print(f"Original rows: {len(df)}")

dirty_mask = df["director"].str.startswith("Director:", na=False)
print(f"Dirty director rows: {dirty_mask.sum()}")

df["director"] = df["director"].str.replace("Director: ", "", regex=False)
df["director"] = df["director"].str.replace("Director:", "", regex=False)

still_dirty = df["director"].str.startswith("Director:", na=False).sum()
print(f"Dirty rows after fix: {still_dirty}")

before = len(df)
df = df.drop_duplicates(subset=["title", "release_year"], keep="first")
after = len(df)
print(f"Removed {before - after} duplicate rows")
print(f"Clean rows: {after}")

df.to_csv("data/movies_clean.csv", index=False, encoding="utf-8")
print(f"\nSaved movies_clean.csv with {len(df)} rows")

print("\nSample director values (should have no 'Director:' prefix):")
print(df["director"].sample(5).tolist())

print("\nDuplicate check after cleaning:")
remaining_dupes = df.duplicated(subset=["title", "release_year"]).sum()
print(f"Remaining duplicates: {remaining_dupes}")