# check_db.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from database import get_stats, get_all_ratings, get_user_history, init_db

init_db()

# Overall stats
print("=== STATS ===")
stats = get_stats()
for key, value in stats.items():
    print(f"  {key}: {value}")

# All ratings
print("\n=== ALL RATINGS ===")
ratings = get_all_ratings()
if ratings:
    for r in ratings:
        print(f"  {r['user_id']} rated '{r['title']}' → {r['rating']} stars at {r['timestamp']}")
else:
    print("  No ratings yet")

# History for a specific user — change name to match what you typed
print("\n=== USER HISTORY ===")
history = get_user_history("your_name_here", limit=10)
if history:
    for h in history:
        print(f"  {h['action']} → {h['title']} (rating: {h['rating']}) at {h['timestamp']}")
else:
    print("  No history found for this user")
