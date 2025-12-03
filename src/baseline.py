import re
from typing import List


# Simple keyword lists per class:
# 0: World, 1: Sports, 2: Business, 3: Sci/Tech
WORLD_KEYWORDS = [
    "world", "president", "government", "minister", "war", "iraq", "UN", "election",
    "europe", "asia", "africa", "country", "diplomat", "parliament", "military",
]
SPORTS_KEYWORDS = [
    "game", "team", "season", "league", "coach", "player", "score", "win", "loss",
    "cup", "tournament", "final", "match", "goal", "nba", "nfl", "mlb", "olympics",
]
BUSINESS_KEYWORDS = [
    "market", "stock", "shares", "company", "profit", "losses", "merger", "bank",
    "deal", "business", "economy", "growth", "revenue", "dollar", "investment",
]
SCITECH_KEYWORDS = [
    "technology", "tech", "computer", "software", "internet", "phone", "science",
    "research", "space", "nasa", "robot", "engineer", "device", "chip", "online",
]

CLASS_KEYWORDS = {
    0: WORLD_KEYWORDS,
    1: SPORTS_KEYWORDS,
    2: BUSINESS_KEYWORDS,
    3: SCITECH_KEYWORDS,
}


def clean_text(text: str) -> str:
    """Basic text cleaning: lowercase, remove extra whitespace."""
    text = text.lower()
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def baseline_predict_one(text: str) -> int:
    """
    Predict AG News label using simple keyword matching.

    Returns class id in {0: world, 1: sports, 2: business, 3: sci/tech}.
    """
    text = clean_text(text)
    scores = {k: 0 for k in CLASS_KEYWORDS.keys()}

    for cls, keywords in CLASS_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                scores[cls] += 1

    # If no keywords matched, default to class 0 (World)
    if all(v == 0 for v in scores.values()):
        return 0

    # Otherwise pick the class with the highest keyword count
    best_class = max(scores.items(), key=lambda kv: kv[1])[0]
    return best_class


def baseline_predict_batch(texts: List[str]) -> List[int]:
    """Apply the baseline classifier to a list of headlines."""
    return [baseline_predict_one(t) for t in texts]
