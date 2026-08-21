import os
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query, Header, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from database import init_db, log_interaction, get_user_history, get_stats
from pydantic import BaseModel, Field
from recommender import Recommender
from expanders.base import BaseExpander
from expanders.ollama_expander import OllamaExpander
from expanders.local_expander import LocalExpander
from expander_model.config import api_config, database_config, expander_config

ACCESS_CODE = os.environ.get(api_config.access_code_env_var)

def verify_access(x_access_code: str = Header(None)):
    if ACCESS_CODE and x_access_code != ACCESS_CODE:
        raise HTTPException(status_code=401, detail="Invalid access code")

recommender: Recommender = None
expander: BaseExpander = None

def _load_expander() -> BaseExpander:

    # set EXPANDER = local to use the model built and trained from scratch
    # the default expander is Ollama 

    mode = expander_config.mode.lower()
    if mode == "local":
        try:
            run_dir = expander_config.local.run_dir or None
            exp = LocalExpander(run_dir=run_dir)
            print("Using local expander")
            return exp
        except FileNotFoundError:
            print("Local model not found... falling back to ollama")
    
    try:
        exp = OllamaExpander()
        print("Using OllamaExpander")
        return exp
    except Exception as e:
        print(f"Ollama not available ({e}) — no expansion")
        return None
    
# Runs at startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    global recommender, expander

    print("Loading reommender...")
    recommender = Recommender()

    print("Loading expander...")
    expander = _load_expander()

    print("Initializing database...")
    init_db()

    print("Ready")
    yield


app = FastAPI(title="Semantic Movie Recommender", description="Find movies by describing what you want.", version="1.0.0", lifespan=lifespan,)
app.add_middleware(CORSMiddleware,
                   allow_origins = api_config.cors_allowed_origins,
                   allow_credentials = True,
                   allow_methods = ["GET", "POST"],
                   allow_headers = ["*"],)

# Response models - pydantix models
class RecommendRequest(BaseModel):
    query:str = Field(..., min_length=3, description="Free-text movie description")
    top_k:int = Field(20, ge=1, le=20)
    genre_filter:str|None = Field(None, description="Filter by genre e.g. 'drama'")
    expand_query:bool = Field(True,  description="Use LLM to expand vague queries")
    search_mode:str = Field("semantic", description = "'semantic' - find by description | 'title' - find by movie name")
    user_id: str|None = Field(None)

class MovieResult(BaseModel):
    rank: int
    title: str
    year: int
    genre: str
    director: str
    cast: str
    plot_summary: str
    score: float
    movie_id: int = 0
    poster_url: str = ""

class RecommendResponse(BaseModel):
    query: str
    expanded_query: str|None
    results: list[MovieResult]
    took_ms: float
    

class SimilarResponse(BaseModel):
    movie_id: int
    title: str
    results: list[MovieResult]

class FeedbackRequest(BaseModel):
    user_id:str = Field(..., min_length=1)
    movie_id:int = Field(...)
    title:str =Field(...)
    action:str = Field(...)
    rating:float = Field(None, ge=1.0, le=10.0)
    query:str = Field(None)

class FeedbackResponse(BaseModel):
    interaction_id: int
    message: str

class HistoryResponse(BaseModel):
    user_id: str
    interactions: list[dict]
    total: int

# End points
@app.post("/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest, _=Depends(verify_access)):
    print(f"FEEDBACK RECEIVED: {req}")
    valid_actions = {"click", "rate", "watch"}
    if req.action not in valid_actions:
        raise HTTPException(status_code=400, detail=f"action must be one of: {valid_actions}")
    if req.action == "rate" and req.rating is None:
        raise HTTPException(status_code=400, detail="rating required when action='rate'")

    interaction_id = log_interaction(
        user_id = req.user_id,
        movie_id = req.movie_id,
        title = req.title,
        action = req.action,
        rating = req.rating,
        query = req.query,
    )
    return FeedbackResponse(
        interaction_id = interaction_id,
        message = f"Logged {req.action} for {req.title}"
    )

@app.get("/history/{user_id}", response_model=HistoryResponse)
def history(
    user_id: str,
    limit: int = Query(default=20, ge=1, le=100),
    action: str = Query(default=None),
    _=Depends(verify_access)
):
    interactions = get_user_history(user_id, limit=limit, action=action)
    return HistoryResponse(user_id=user_id, interactions=interactions, total=len(interactions))
    
@app.get("/health")
def health(_=Depends(verify_access)):
    return{"status": "ok",
           "movie_indexed": len(recommender.movies) if recommender else 0,
           "expander": type(expander).__name__ if expander else "None",}

@app.post("/recommend", response_model = RecommendResponse)
def recommend(req: RecommendRequest, _=Depends(verify_access)):
    start = time.time()

    # validate search_mode
    valid_modes = {"semantic", "title"}
    if req.search_mode not in valid_modes:
        raise HTTPException(status_code = 400, detail = f"search mode must be on of {valid_modes}")
    
    # optionally expand the query
    expanded_query = None
    search_query = req.query

    # title
    if req.search_mode == "title":
        results = recommender.search_by_title(req.query, top_k=req.top_k)
        took_ms = (time.time() - start) * 1000
        return RecommendResponse(query=req.query, expanded_query=None,
                                results=[MovieResult(
                                        rank = r.rank,
                                        title= r.title,
                                        year= r.year,
                                        genre= r.genre,
                                        director= r.director,
                                        cast=r.cast,
                                        plot_summary= r.plot_summary[:300],
                                        score= r.score,
                                        movie_id = r.movie_id,
                                        poster_url = r.poster_url)
                                    
                                    for r in results                    
                                ],
                                took_ms= round(took_ms, 2),
                                ) 
    
    # semantic search
    if req.expand_query and expander is not None:
        expanded = expander.expand(req.query)
        if expanded != req.query:
            expanded_query = expanded
            search_query = expanded
    
    print(f"DEBUG: searching for: {search_query[:50]}")

    # search
    results =recommender.search(query=search_query, top_k=req.top_k, genre_filter=req.genre_filter)
    print(f"DEBUG: got {len(results)} results, first score: {results[0].score if results else 'NO RESULTS'}")
    print(f"DEBUG: first title: {results[0].title if results else 'NO RESULTS'}")

    took_ms = (time.time() - start) * 1000
    return RecommendResponse(query=req.query, expanded_query=expanded_query,
                             results=[MovieResult(
                                    rank = r.rank,
                                    title= r.title,
                                    year= r.year,
                                    genre= r.genre,
                                    director= r.director,
                                    cast = r.cast,
                                    plot_summary= r.plot_summary[:300],
                                    score= r.score,
                                    movie_id = r.movie_id,
                                    poster_url = r.poster_url)

                                for r in results
                             ],
                             took_ms= round(took_ms, 2),
                             )


@app.get("/similar/{movie_id}", response_model = SimilarResponse)
def similar(movie_id: int, top_k: int = Query(default=5, ge=1, le=20)):
    try:
        results = recommender.similar_to(movie_id, top_k=top_k)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"movie_id {movie_id} not found")

    # Get the title of the source movie
    movie_row = recommender.movies[recommender.movies["movie_id"] == movie_id]
    title     = movie_row.iloc[0]["title"] if len(movie_row) > 0 else "Unknown"

    return SimilarResponse(movie_id = movie_id, title = title,
                           results=[MovieResult(
                                    rank = r.rank,
                                    title= r.title,
                                    year= r.year,
                                    genre= r.genre,
                                    director= r.director,
                                    cast = r.cast,
                                    plot_summary= r.plot_summary[:300],
                                    score= r.score,
                                    movie_id = r.movie_id,
                                    poster_url = r.poster_url)
                                for r in results
                             ],)


@app.get("/movies/search")
def search_movies(
    title: str = Query(..., min_length=2, description="Movie title to search"),
    limit: int = Query(default=10, ge=1, le=50),
):
    # search movies by title
    mask    = recommender.movies["title"].str.contains(title, case=False, na=False)
    matches = recommender.movies[mask].head(limit)

    if len(matches) == 0:
        raise HTTPException(status_code=404, detail=f"No movies found matching '{title}'")

    return {
        "query":   title,
        "results": [
            {
                "movie_id": int(row["movie_id"]),
                "title": row["title"],
                "year": int(row["release_year"]),
                "genre": row["genre"],
                "director": row["director"],
                "cast": row["cast"]
            }
            for _, row in matches.iterrows()
        ]
    }

@app.get("/debug")
def debug():
    import numpy as np
    scores_sample = (recommender.embeddings @ recommender.embeddings[0]).tolist()[:5]
    return {
        "embeddings_shape": list(recommender.embeddings.shape),
        "movies_columns": recommender.movies.columns.tolist(),
        "has_movie_id": "movie_id" in recommender.movies.columns,
        "movies_count": len(recommender.movies),
        "sample_scores": scores_sample,
        "first_movie": recommender.movies.iloc[0]["title"],
    }

@app.get("/debug2")
def debug2():
    import numpy as np
    query = "action love movie"
    query_embed = recommender.model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    )[0]
    scores = recommender.embeddings @ query_embed
    top_idx = np.argpartition(scores, -5)[-5:]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return {
        "top_scores": scores[top_idx].tolist(),
        "top_titles": recommender.movies.iloc[top_idx]["title"].tolist(),
        "top_idx": top_idx.tolist(),
    }

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")
