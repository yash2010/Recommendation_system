# 🎬 Semantic Movie Recommendation System

A movie recommendation system that finds films based on natural language descriptions rather than keyword matching. Built from scratch using transformer embeddings, semantic similarity search, and an optional LLM query expansion layer, all served through a REST API.

---

## ✨ Features

- **Semantic search** - finds movies by meaning, not just keywords
- **Query expansion** - automatically enriches vague queries using a local LLM (Ollama)
- **Custom transformer** - seq2seq query expander built from scratch in PyTorch
- **Movies** indexed from Wikipedia
- **REST API** - FastAPI with interactive Swagger docs
- **Two expander modes** - Ollama (high quality) or local trained model (offline)
- **Similar movies** - find movies similar to any movie by ID
- **Genre filtering** - filter results by genre

---

## 🏗️ Architecture

```
User types a description
        │
Query Expansion (Ollama / Custom Transformer)
  "a dark thriller" -> "A psychologically intense thriller
                        featuring an unreliable narrator..."
        │
Sentence Transformer Embedding (all-MiniLM-L6-v2)
  Text -> dimensional vector
        │
Cosine Similarity Search
  Query vector · Movie vectors 
        │
Top-k Results via REST API
```

---

## 🚀 Quick Start

### 1. Clone and install

```bash
git clone https://github.com/yash2010/movie_recommendation_system.git
cd movie_recommendation_system
```

Install dependencies using 

```bash
conda env create -f environment.yml
```


### 2. Get the dataset

This project uses the **Wikipedia Movie Plots** dataset by Justin Robischon.
- **Source:** [Kaggle - Wikipedia Movie Plots](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots)
- **Size:** ~35,000 movies with full Wikipedia plot descriptions
- **Columns:** Release Year, Title, Origin/Ethnicity, Director, Cast, Genre, Wiki Page, Plot, PlotSummary

Download the Wikipedia Movie Plots dataset and place it in `data/`:

The CSV should have these columns:
`release_year, title, origin_ethnicity, director, cast, genre, wiki_page, plot, plotsummary`

### 3. Clean the data

```bash
python clean_data.py
```
The cleaned dataset will be stored under the path:
```
data/movies_clean.csv
```
### 4. Build the embedding index

This embeds all the movies using sentence-transformers.

```bash
python build_index.py
```

### 5. Start Ollama (for query expansion)

```bash
# Install Ollama from https://ollama.com
ollama pull llama3
ollama serve
```

### 6. Start the API

```bash
uvicorn api:app --reload
```

Visit `http://localhost:8000/docs` for interactive API documentation.

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
    "expand_query": true
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
      "score": 0.5843
    }
  ],
  "took_ms": 68.06
}
```

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

### `GET /health`

Check system status.

```bash
curl http://localhost:8000/health
```

---

## 🔧 Configuration

### Switch between expanders

```bash
# Use Ollama (default, best quality)
uvicorn api:app --reload

# Use local trained model (offline, faster)
set EXPANDER=local  # Windows
export EXPANDER=local  # Mac/Linux
uvicorn api:app --reload
```

### Train the custom query expander

```bash
# Generate training data (requires Ollama)
python generate_training_data.py

# Train the model
python train.py
```

---

## 🧠 Custom Transformer

The query expander is a seq2seq transformer built entirely from scratch in PyTorch - no pretrained weights, no Hugging Face models.

## 📊 How Semantic Search Works

Traditional keyword search matches exact words. Semantic search matches meaning:

| Query | Keyword Search | Semantic Search |
|---|---|---|
| "dark thriller" | Movies containing "dark" and "thriller" | Movies with themes of dread, tension, moral ambiguity |
| "film that makes you think" | No matches (too vague) | Philosophical dramas, thought-provoking sci-fi |
| "something like Inception" | No matches | Mind-bending sci-fi, non-linear narratives |

The system encodes both movies and queries as a dimensional vectors using `all-MiniLM-L6-v2`. Cosine similarity between vectors measures semantic relatedness. Higher score = more similar meaning.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [sentence-transformers](https://www.sbert.net/) for the embedding model
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [Ollama](https://ollama.com/) for local LLM inference
- [Kaggle](https://www.kaggle.com/) for movie dataset

## Note

⚠️**Work in Progress**

This recommendation system is till under development and is currently under development and is not yet fully complete. At the moment, only the semantic search functionality has been implemented. Additional features, enhancements, and optimizations are planned and will be introduced in future updates.
