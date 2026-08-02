from expanders.ollama_expander import OllamaExpander
from expanders.local_expander import LocalExpander

queries = [
    "a dark psychological thriller",
    "something for a rainy day",
    "a film that makes you think",
    "science fiction about artificial intelligence",  # specific — won't expand
]

# ── Test Ollama ───────────────────────────────────────────────────────────────
print("=" * 60)
print("OLLAMA EXPANDER")
print("=" * 60)
ollama_exp = OllamaExpander(model="llama3")

for q in queries:
    expanded = ollama_exp.expand(q)
    print(f"\nQuery:    {q}")
    print(f"Expanded: {expanded[:120]}...")

# ── Test Local ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("LOCAL EXPANDER")
print("=" * 60)
local_exp = LocalExpander()

for q in queries:
    expanded = local_exp.expand(q)
    print(f"\nQuery:    {q}")
    print(f"Expanded: {expanded[:120]}")