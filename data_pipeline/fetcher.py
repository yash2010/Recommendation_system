import gzip
import json
import time
import requests
from datetime import datetime, timedelta
from data_pipeline.config import tmdb_config


class TMDBFetcher:

    def __init__(self, config=None):
        self.config  = config or tmdb_config
        self.headers = {
            "Authorization": f"Bearer {self.config.token}",
            "accept":        "application/json",
        }
        self._validate()

    def _validate(self):
        if not self.config.token:
            raise ValueError(
                "TMDB_TOKEN not set.\n"
                "Add this to your .env file:\n"
                "TMDB_TOKEN=your_token_here"
            )

    def get(self, endpoint: str, params: dict = {}, retry: int = 0) -> dict | None:
        try:
            res = requests.get(
                f"{self.config.base_url}{endpoint}",
                headers = self.headers,   
                params  = params,             
                timeout = self.config.timeout,
            )

            if res.status_code == 200:
                return res.json()
            elif res.status_code == 429:
                wait = int(res.headers.get("Retry-After", 10))
                print(f"  Rate limited — waiting {wait}s...")
                time.sleep(wait)
                return self.get(endpoint, params, retry)
            elif res.status_code in (500, 502, 503, 504):
                if retry < self.config.max_retries:
                    time.sleep(2 ** retry)
                    return self.get(endpoint, params, retry + 1)
                return None
            elif res.status_code == 404:
                return None
            else:
                return None

        except requests.exceptions.Timeout:
            if retry < self.config.max_retries:
                time.sleep(2)
                return self.get(endpoint, params, retry + 1)
            return None
        except requests.exceptions.ConnectionError:
            if retry < self.config.max_retries:
                time.sleep(5)
                return self.get(endpoint, params, retry + 1)
            return None
        except Exception as e:
            print(f"  Unexpected error: {e}")
            return None

    def discover_page(self, page: int, language: str = None) -> list[dict]:
        params = {
            "sort_by":        self.config.sort_by,
            "page":           page,
            "vote_count.gte": self.config.min_votes,
            "popularity.gte": self.config.min_popularity,
        }
        if language:
            params["with_original_language"] = language

        data = self.get("/discover/movie", params=params)
        return data.get("results", []) if data else []

    def movie_details(self, movie_id: int) -> dict | None:
        return self.get(
            f"/movie/{movie_id}",
            params={"append_to_response": "credits,translations"},
        )

    def export_ids(self, date_str: str = None) -> list[int]:
        if not date_str:
            yesterday = datetime.now() - timedelta(days=1)
            date_str  = yesterday.strftime("%m_%d_%Y")

        url = f"https://files.tmdb.org/p/exports/movie_ids_{date_str}.json.gz"
        print(f"Downloading export file: {url}")

        try:
            res = requests.get(url, timeout=120, stream=True)

            if res.status_code != 200:
                day_before = datetime.now() - timedelta(days=2)
                date_str   = day_before.strftime("%m_%d_%Y")
                url        = f"https://files.tmdb.org/p/exports/movie_ids_{date_str}.json.gz"
                print(f"Trying previous day: {url}")
                res        = requests.get(url, timeout=120, stream=True)
                if res.status_code != 200:
                    print("Export file not available — falling back to discover")
                    return []

            ids = []
            with gzip.GzipFile(fileobj=res.raw) as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        if obj.get("popularity", 0) >= self.config.export_min_popularity:
                            ids.append(obj["id"])
                    except:
                        continue

            print(f"Export: {len(ids):,} movies after popularity filter")
            if len(ids) > self.config.export_max_movies:
                ids = ids[:self.config.export_max_movies]
            return ids

        except Exception as e:
            print(f"Export download failed: {e}")
            return []
