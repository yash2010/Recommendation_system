# 🎬 Semantic Movie Recommendation System

A movie recommendation system that finds films based on natural language descriptions rather than keyword matching. Built from scratch using transformer embeddings, semantic similarity search, and an optional LLM query expansion layer, all served through a REST API with a web UI on top.

---

## ✨ Features

- **Semantic search** - finds movies by meaning, not just keywords
- **Query expansion** - enriches vague queries via Ollama or a from-scratch trained transformer
- **Custom transformer** - seq2seq query expander (encoder/decoder + attention) built from scratch in PyTorch, with a choice of tokenizers (word-level, BPE via HuggingFace, or BPE from scratch)
- **TMDB-backed data pipeline** - fetches, processes, and checkpoints movie data from The Movie Database API across multiple languages
- **SQLite-backed storage** - movies, users, and interactions all live in one database
- **User feedback loop** - logs clicks/ratings/watches per user and exposes history/stats for future personalization
- **REST API** - FastAPI with interactive Swagger docs, optional access-code auth
- **Web UI** - a static front end (`static/index.html`) served directly by the API
- **Similar movies** - find movies similar to any movie by ID
- **Title & genre search** - look movies up by title or filter results by genre

---

## 🏗️ Architecture

```
TMDB API
   │  (data_pipeline: fetcher -> processor -> exporter)
   ▼
data/tmdb_movies.csv  ──►  scripts/csv2db.py  ──►  data/movies.db (SQLite)
   │                                                   │
   │  scripts/build_index.py                           │  users, interactions
   ▼                                                   ▼
artifacts/embeddings.npy (all-MiniLM-L6-v2)      feedback / history endpoints

User types a description
        │
        ▼
Query Expansion (Ollama / custom transformer)
  "a dark thriller" -> "A psychologically intense thriller
                        featuring an unreliable narrator..."
        │
        ▼
Sentence Transformer Embedding (all-MiniLM-L6-v2)
        │
        ▼
Cosine Similarity Search over artifacts/embeddings.npy
        │
        ▼
Top-k Results via REST API / Web UI
```

---

## 📁 Project Layout

```
api.py                  FastAPI app - endpoints, lifespan startup
recommender.py           Loads embeddings + movie DB, does similarity search
database.py              SQLite access - users, interactions, feedback, stats
expander_train.py        Trains the from-scratch query-expander transformer

config/
  api.yaml               Server, database, and expander-mode settings
  expander_train.yaml    Custom transformer architecture + training hyperparams

data_pipeline/           TMDB fetch -> process -> export pipeline
expander_model/          Custom transformer (attention, encoder/decoder, tokenizers)
expanders/               Pluggable query expanders (Ollama, local) behind a common base class
scripts/                 One-off / operational scripts (fetch, migrate, build index, train, inspect)
static/                  Web UI (index.html) + Privacy/Terms pages
```

---

## 🚀 Quick Start

### 1. Clone and install

```bash
git clone https://github.com/yash2010/movie_recommendation_system.git
cd movie_recommendation_system
```

Install dependencies with conda:

```bash
conda env create -f environment.yml
```

or with pip:

```bash
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```
TMDB_TOKEN=your_tmdb_api_read_token
ACCESS_CODE=optional_shared_secret_for_the_api
```

`TMDB_TOKEN` is required to fetch data from TMDB. `ACCESS_CODE` is optional - if set, every request must include an `X-Access-Code` header matching it (see `config/api.yaml`).

### 3. Fetch movie data from TMDB

```bash
python scripts/fetch_tmdb.py --pages 500 --languages en,ta,ko
```

This calls the TMDB discover/details endpoints, checkpoints progress to `data/tmdb_movies.csv`, and skips movie IDs it has already fetched on subsequent runs. Pipeline behavior (rate limiting, retries, min votes/popularity, output paths) is controlled by `data_pipeline/config.yaml`.

### 4. Migrate data into SQLite

```bash
python scripts/csv2db.py
```

Loads `data/tmdb_movies.csv` (or `data/movies_final.parquet`) into `data/movies.db`, and creates the `users`/`interactions` tables used for feedback logging.

### 5. Build the embedding index

```bash
python scripts/build_index.py
```

Embeds every movie with `sentence-transformers/all-MiniLM-L6-v2` and saves `artifacts/embeddings.npy` + `artifacts/movies.parquet`.

### 6. (Optional) Start Ollama for query expansion

```bash
# Install Ollama from https://ollama.com
ollama pull llama3.2:1b
ollama serve
```

### 7. Start the API

```bash
uvicorn api:app --reload
```

Visit `http://localhost:8000/` for the web UI, or `http://localhost:8000/docs` for interactive API documentation.

---

## API Endpoints

### `POST /recommend`

Find movies matching a natural language description.

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "query": "a dark psychological thriller",
    "top_k": 5,
    "genre_filter": "thriller",
    "expand_query": true,
    "search_mode": "semantic"
  }'
```

**Response:**
```json
{
  "query": "a dark psychological thriller",
  "expanded_query": "A psychologically intense thriller featuring...",
  "results": [
    {
      "rank": 1,
      "title": "After the Dark",
      "year": 2014,
      "genre": "thriller",
      "director": "John Huddles",
      "plot_summary": "...",
      "score": 0.5843,
      "movie_id": 15853
    }
  ],
  "took_ms": 68.06
}
```

Set `"search_mode": "title"` to look a movie up by name instead of by description.

### `GET /similar/{movie_id}`

Find movies similar to a given movie.

```bash
curl http://localhost:8000/similar/15853?top_k=5
```

### `GET /movies/search`

Search movies by title to find `movie_id`.

```bash
curl "http://localhost:8000/movies/search?title=inception"
```

### `POST /feedback`

Log a user interaction (`click`, `rate`, or `watch`) for future personalization.

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"user_id": "u1", "movie_id": 15853, "title": "After the Dark", "action": "rate", "rating": 4.5}'
```

### `GET /history/{user_id}`

Fetch a user's logged interactions.

```bash
curl "http://localhost:8000/history/u1?limit=20"
```

### `GET /health`

Check system status (movies indexed, active expander).

```bash
curl http://localhost:8000/health
```

If `ACCESS_CODE` is set in `.env`, all requests above must also include `-H "X-Access-Code: <your code>"`.

---

## 🔧 Configuration

All runtime settings live in `config/*.yaml` (loaded via `expander_model/config.py` and `data_pipeline/config.py`), not hardcoded constants:

- **`config/api.yaml`** - CORS origins, DB path, and `expander.mode` (`ollama` or `local`)
- **`config/expander_train.yaml`** - custom transformer architecture, tokenizer choice, and training hyperparameters
- **`data_pipeline/config.yaml`** - TMDB fetch settings (languages, rate limiting, min votes/popularity, checkpointing)

### Switch expander mode

Edit `config/api.yaml`:

```yaml
expander:
  mode: local   # or: ollama
```

If `local` is selected but no trained checkpoint exists, the API falls back to Ollama automatically.

### Train the custom query expander

```bash
# Generate training pairs (requires Ollama)
python scripts/expanderTraining_data.py

# Train
python expander_train.py
```

Runs are saved under `artifacts/expander/runs/<timestamp>_...`; `LocalExpander` auto-selects the run with the lowest validation loss.

---

## 🧠 Custom Transformer

The from-scratch query expander (`expander_model/`) is a seq2seq transformer - encoder, decoder, multi-head attention, and transformer blocks all implemented directly in PyTorch, no pretrained weights. It supports three interchangeable tokenizers (`word_level`, `bpe_library`, `bpe_scratch`), selected in `config/expander_train.yaml`.

## 📊 How Semantic Search Works

Traditional keyword search matches exact words. Semantic search matches meaning:

| Query | Keyword Search | Semantic Search |
|---|---|---|
| "dark thriller" | Movies containing "dark" and "thriller" | Movies with themes of dread, tension, moral ambiguity |
| "film that makes you think" | No matches (too vague) | Philosophical dramas, thought-provoking sci-fi |
| "something like Inception" | No matches | Mind-bending sci-fi, non-linear narratives |

Movies and queries are both encoded as vectors with `all-MiniLM-L6-v2`. Cosine similarity between vectors measures semantic relatedness — higher score = more similar meaning.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [sentence-transformers](https://www.sbert.net/) for the embedding model
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [Ollama](https://ollama.com/) for local LLM inference
- [The Movie Database (TMDB)](https://www.themoviedb.org/) for movie data
- [Figma](https://www.figma.com/) for designing the web UI
- [Claude design](https://claude.ai/design) for building the web UI

## 🗺️ Planned Features

⚠️ **Work in Progress** - this project is under active development.

- **Collaborative filtering** - use logged clicks/ratings/watches (`/feedback`, `/history`) to power personalized recommendations, instead of just storing them
- **Custom retriever transformer** - a bi-encoder built from scratch to replace `all-MiniLM-L6-v2` as the embedding model
- **UI filters** - language and genre filter controls in the web UI (genre filtering already exists in the API, just not exposed in the frontend)
- **Fine-tuned FLAN-T5 expander** - `expanders/finetuned_expander.py` exists, but no model has been trained yet and it isn't wired into `config/api.yaml`'s expander-mode switch

