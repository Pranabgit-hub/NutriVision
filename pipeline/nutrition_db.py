"""
Nutrition database + retrieval layer.

Design note: this is deliberately NOT a generative RAG step. Matching a VLM
label like "fried rice with vegetables" to a canonical DB entry is a retrieval
problem (nearest-neighbor over names), and once we have a match, computing
macros for a given mass is pure arithmetic. There is no reason to let an LLM
"generate" a calorie count -- that would trade a deterministic computation for
a hallucination-prone one. Swap the fuzzy-match retriever below for an
embedding-based retriever (e.g. sentence-transformers + FAISS) if the label
vocabulary grows beyond what edit-distance matching handles well -- see
`EmbeddingRetriever` stub at the bottom for the extension point.
"""
from __future__ import annotations

import csv
import dataclasses
from pathlib import Path
from typing import Optional

from rapidfuzz import process, fuzz

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "nutrition_db.csv"


@dataclasses.dataclass
class FoodEntry:
    food_id: str
    name: str
    category: str
    kcal_per_100g: float
    protein_g: float
    fat_g: float
    carbs_g: float
    fiber_g: float
    sugar_g: float
    sodium_mg: float
    iron_mg: float
    calcium_mg: float
    vitamin_c_mg: float
    density_g_per_ml: float

    def scale(self, grams: float) -> "MacroResult":
        factor = grams / 100.0
        return MacroResult(
            food_name=self.name,
            grams=grams,
            kcal=self.kcal_per_100g * factor,
            protein_g=self.protein_g * factor,
            fat_g=self.fat_g * factor,
            carbs_g=self.carbs_g * factor,
            fiber_g=self.fiber_g * factor,
            sugar_g=self.sugar_g * factor,
            sodium_mg=self.sodium_mg * factor,
            iron_mg=self.iron_mg * factor,
            calcium_mg=self.calcium_mg * factor,
            vitamin_c_mg=self.vitamin_c_mg * factor,
        )


@dataclasses.dataclass
class MacroResult:
    food_name: str
    grams: float
    kcal: float
    protein_g: float
    fat_g: float
    carbs_g: float
    fiber_g: float
    sugar_g: float
    sodium_mg: float
    iron_mg: float
    calcium_mg: float
    vitamin_c_mg: float

    def __add__(self, other: "MacroResult") -> "MacroResult":
        if not isinstance(other, MacroResult):
            return NotImplemented
        return MacroResult(
            food_name="combined",
            grams=self.grams + other.grams,
            kcal=self.kcal + other.kcal,
            protein_g=self.protein_g + other.protein_g,
            fat_g=self.fat_g + other.fat_g,
            carbs_g=self.carbs_g + other.carbs_g,
            fiber_g=self.fiber_g + other.fiber_g,
            sugar_g=self.sugar_g + other.sugar_g,
            sodium_mg=self.sodium_mg + other.sodium_mg,
            iron_mg=self.iron_mg + other.iron_mg,
            calcium_mg=self.calcium_mg + other.calcium_mg,
            vitamin_c_mg=self.vitamin_c_mg + other.vitamin_c_mg,
        )


class NutritionDB:
    """Loads the seed CSV and does fuzzy name retrieval.

    Swap `DEFAULT_DB_PATH` for USDA FoodData Central / IFCT exports to scale
    coverage -- the CSV schema here is a minimal subset of what those sources
    provide; extend `FoodEntry` as needed.
    """

    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.entries: list[FoodEntry] = []
        self._load(db_path)
        self._names = [e.name for e in self.entries]

    def _load(self, db_path: Path) -> None:
        with open(db_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.entries.append(
                    FoodEntry(
                        food_id=row["food_id"],
                        name=row["name"],
                        category=row["category"],
                        kcal_per_100g=float(row["kcal_per_100g"]),
                        protein_g=float(row["protein_g"]),
                        fat_g=float(row["fat_g"]),
                        carbs_g=float(row["carbs_g"]),
                        fiber_g=float(row["fiber_g"]),
                        sugar_g=float(row["sugar_g"]),
                        sodium_mg=float(row["sodium_mg"]),
                        iron_mg=float(row["iron_mg"]),
                        calcium_mg=float(row["calcium_mg"]),
                        vitamin_c_mg=float(row["vitamin_c_mg"]),
                        density_g_per_ml=float(row["density_g_per_ml"]),
                    )
                )

    def match(self, label: str, top_k: int = 3, score_cutoff: float = 45.0):
        """Return up to top_k (FoodEntry, score) candidates for a free-text label.

        score_cutoff is on rapidfuzz's 0-100 scale. Callers (or the human
        confirmation UI) should treat scores below ~70 as "needs user
        confirmation" rather than auto-accepting the top match.
        """
        results = process.extract(
            label, self._names, scorer=fuzz.WRatio, limit=top_k, score_cutoff=score_cutoff
        )
        out = []
        for name, score, idx in results:
            out.append((self.entries[idx], score))
        return out

    def best_match(self, label: str) -> Optional[tuple[FoodEntry, float]]:
        matches = self.match(label, top_k=1)
        return matches[0] if matches else None


class EmbeddingRetriever:
    """Extension point: swap fuzzy string matching for semantic retrieval.

    Not implemented here because it requires downloading embedding-model
    weights (network access to a model hub), which this environment doesn't
    have. To enable: embed `self._names` once at load time, embed the query
    label, and do cosine-similarity top-k instead of `process.extract`. The
    rest of the pipeline (MacroResult, aggregation) is unaffected -- this is
    purely a retrieval-quality upgrade.
    """

    def __init__(self, *_args, **_kwargs):
        raise NotImplementedError(
            "Plug in sentence-transformers/OpenAI/Voyage embeddings + FAISS here "
            "once you have model-hub network access. NutritionDB.match() is a "
            "drop-in fuzzy-match baseline until then."
        )
