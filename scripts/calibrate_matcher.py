"""Calibración del fallback por embeddings del category matcher (E1-21, D-13).

Términos que el léxico no resuelve: positivos (con su categoría esperada) y negativos (fuera
del vocabulario). Para cada modelo guarda el mejor match y su similitud, y el resultado con el
umbral de la config. Uso:
    python scripts/calibrate_matcher.py [--models <hf_id> ...] [--out outputs/matcher]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scene.config import load_config
from scene.io.dataset import load_vocabulary
from scene.qa.category_matcher import CategoryMatcher, HFTextEmbedder, matcher_from_config

POSITIVES = {
    "mug": "cup",
    "coffee mug": "cup",
    "tacita": "cup",
    "teléfono inteligente": "cell phone",
    "smartphone": "cell phone",
    "auto deportivo": "car",
    "camioneta": "car",
    "pantalla de televisión": "television set",
    "perrito": "dog",
    "gatito": "cat",
    "bicicletas de montaña": "bicycle",
    "copa de vino": "drinking glass",
    "computadora": "laptop",
    "heladerita": "refrigerator",
    "platos hondos": "plate",
    "sillón": "sofa",
    "gorro": "hat",
    "zapatillas deportivas": "shoe",
}
NEGATIVES = [
    "dinosaurio", "piano", "nube", "árbol", "edificio", "cielo", "montaña", "ventana", "puerta",
    "cuadro", "pared", "agua", "león", "tigre", "guitarra", "pelota", "hora", "color",
    "escalera", "calle", "cuchara", "sartén",
]  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", nargs="+", default=["configs/base.yaml"])
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--out", default="outputs/matcher")
    args = parser.parse_args()

    cfg = load_config(*args.config)
    ecfg = cfg["qa"]["matcher"]["embeddings"]
    threshold = ecfg["threshold"]
    categories = load_vocabulary(cfg)
    exact = matcher_from_config(cfg, categories)
    exact.embedder = None  # solo léxico: para descartar términos que ya resuelve

    rows = []
    for model_id in args.models or [ecfg["model_id"]]:
        matcher = CategoryMatcher(
            categories,
            lexicon=exact.lexicon,
            synonyms=exact.synonyms,
            embedder=HFTextEmbedder(model_id, batch_size=ecfg["batch_size"]),
            embedding_threshold=0.0,
        )
        terms = [(t, c) for t, c in POSITIVES.items()] + [(t, None) for t in NEGATIVES]
        for term, expected in terms:
            if exact.match(term) is not None:
                continue
            m = matcher.match(term)
            accepted = m.score >= threshold
            if expected is None:
                outcome = "false_positive" if accepted else "true_negative"
            elif not accepted:
                outcome = "missed"
            else:
                outcome = "correct" if m.category_name == expected else "wrong_category"
            rows.append(
                {
                    "model_id": model_id,
                    "term": term,
                    "expected": expected,
                    "best_category": m.category_name,
                    "best_text": m.matched_text,
                    "similarity": round(m.score, 4),
                    "outcome": outcome,
                }
            )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "matcher_calibration.csv", index=False)
    summary = {
        model_id: {
            "threshold": threshold,
            "max_negative_similarity": float(g[g.expected.isna()].similarity.max()),
            **g.outcome.value_counts().to_dict(),
        }
        for model_id, g in df.groupby("model_id")
    }
    (out / "matcher_calibration.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
