import ollama
from expanders.base import BaseExpander
from model.config import SYSTEM_PROMPT


class OllamaExpander(BaseExpander):

    def __init__(self, model: str = "llama3"):
        self.model = model
        self._verify_connection()

    def _verify_connection(self):
        try:
            models = ollama.list()
            available = [m.model for m in models.get("models", [])]
            if not any(self.model in m for m in available):
                print(f"Warning: {self.model} not found.")
        except Exception as e:
            print(f"Warning: Ollama not reachable — {e}")
            print("Make sure Ollama is running.")

    def expand(self, query: str) -> str:
        """Expand query using local Ollama model."""
        if not self.should_expand(query):
            return query

        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": f"Expand this vague movie query:\n{query}"}
                ]
            )
            expanded = response["message"]["content"].strip()
            return expanded if expanded else query

        except Exception as e:
            print(f"Ollama expansion failed: {e}")
            return query  
