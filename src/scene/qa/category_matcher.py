"""Category matcher: texto (inglés o español) → category_id del vocabulario, o `None`.

Orden de búsqueda (PLAN §4.5): para cada forma del texto (tal cual y en singular) se prueba
nombre del vocabulario → léxico español → sinónimos LVIS. Si nada coincide, como último recurso,
similitud de embeddings con umbral. Sin match → `None` (la pregunta queda `unsupported`).
Solo usa el vocabulario y recursos públicos, nunca anotaciones.
"""

from __future__ import annotations

import itertools
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from scene.detection.vocabulary import clean_name, load_lvis_synonyms
from scene.io.dataset import Category

# Una función que recibe textos y devuelve sus embeddings normalizados (una fila por texto).
Embedder = Callable[[list[str]], np.ndarray]

LEADING_ARTICLES = {"the", "a", "an", "el", "la", "los", "las", "un", "una", "unos", "unas"}
IRREGULAR_PLURALS = {
    "people": "person",
    "men": "man",
    "women": "woman",
    "children": "child",
    "mice": "mouse",
    "feet": "foot",
    "teeth": "tooth",
}
# Sufijo de plural → reemplazos posibles (inglés y español). Se prueban todos; solo vale el
# que coincide con alguna entrada del índice, así que generar formas de más no rompe nada.
PLURAL_SUFFIXES = (
    ("ies", ("y",)),
    ("ves", ("f", "fe")),
    ("ces", ("z",)),
    ("es", ("",)),
    ("s", ("",)),
)
MIN_WORD_FOR_SINGULAR = 4
MAX_CANDIDATES = 64


def normalize_text(text: str) -> str:
    """Minúsculas, sin acentos (ñ → n) ni signos, espacios simples y sin artículo inicial."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    words = re.sub(r"[^a-z0-9]+", " ", text).split()
    if len(words) > 1 and words[0] in LEADING_ARTICLES:
        words = words[1:]
    return " ".join(words)


def _singular_options(word: str) -> list[str]:
    """La palabra tal cual y sus posibles singulares."""
    options = [word]
    if word in IRREGULAR_PLURALS:
        options.append(IRREGULAR_PLURALS[word])
    elif len(word) >= MIN_WORD_FOR_SINGULAR:
        for suffix, replacements in PLURAL_SUFFIXES:
            if word.endswith(suffix):
                options += [word[: -len(suffix)] + r for r in replacements]
    return list(dict.fromkeys(options))


def singular_candidates(text: str) -> list[str]:
    """Formas a buscar: primero el texto tal cual, después combinaciones con palabras en singular.

    Se combinan todas las palabras porque en español el plural puede ir en cualquiera
    ("raquetas de tenis", "teléfonos celulares").
    """
    words = normalize_text(text).split()
    if not words:
        return []
    combos = itertools.product(*(_singular_options(w) for w in words))
    return [" ".join(c) for c in itertools.islice(combos, MAX_CANDIDATES)]


@dataclass(frozen=True)
class CategoryMatch:
    category_id: int
    category_name: str
    method: str  # name | lexicon_es | lvis_synonym | embedding
    score: float  # 1.0 en los matches exactos; similitud coseno en `embedding`
    matched_text: str  # entrada del índice que coincidió


@dataclass
class CategoryMatcher:
    """Índices por nivel de confianza; dentro de un nivel, un texto ambiguo se descarta."""

    categories: list[Category]
    lexicon: dict[str, list[str]] = field(default_factory=dict)  # nombre en inglés → términos
    synonyms: dict[int, list[str]] = field(default_factory=dict)  # category_id → sinónimos LVIS
    embedder: Embedder | None = None
    embedding_threshold: float = 1.0
    # Textos que apuntaban a más de una categoría dentro de un nivel (para trazabilidad).
    dropped: dict[str, list[int]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._names = {c.id: c.name for c in self.categories}
        by_name = {normalize_text(clean_name(c.name)): c.id for c in self.categories}
        unknown = [n for n in self.lexicon if normalize_text(n) not in by_name]
        if unknown:
            raise ValueError(
                f"El léxico tiene categorías que no están en el vocabulario: {unknown}"
            )

        raw_tiers = {
            "name": [(c.name, c.id) for c in self.categories],
            "lexicon_es": [
                (term, by_name[normalize_text(name)])
                for name, terms in self.lexicon.items()
                for term in terms
            ],
            "lvis_synonym": [(syn, cid) for cid, syns in self.synonyms.items() for syn in syns],
        }
        self._tiers: list[tuple[str, dict[str, int]]] = []
        self._display: dict[str, str] = {}  # clave normalizada → texto original (embeddings)
        for method, entries in raw_tiers.items():
            owners: dict[str, set[int]] = {}
            for text, cid in entries:
                key = normalize_text(clean_name(text))
                if key:
                    owners.setdefault(key, set()).add(cid)
                    self._display.setdefault(key, clean_name(text))
            index = {k: next(iter(ids)) for k, ids in owners.items() if len(ids) == 1}
            for k, ids in owners.items():
                if len(ids) > 1:
                    self.dropped[f"{method}:{k}"] = sorted(ids)
            self._tiers.append((method, index))
        self._emb_keys: list[tuple[str, int]] | None = None
        self._emb_matrix: np.ndarray | None = None

    def match(self, text: str) -> CategoryMatch | None:
        """Categoría para `text`, o `None` si no hay match confiable."""
        for candidate in singular_candidates(text):
            for method, index in self._tiers:
                if candidate in index:
                    cid = index[candidate]
                    return CategoryMatch(cid, self._names[cid], method, 1.0, candidate)
        return self._match_embedding(text)

    def match_id(self, text: str) -> int | None:
        found = self.match(text)
        return found.category_id if found else None

    def _match_embedding(self, text: str) -> CategoryMatch | None:
        if self.embedder is None or not normalize_text(text):
            return None
        if self._emb_matrix is None:
            # Una entrada por clave: la del nivel más confiable donde aparece.
            seen: dict[str, int] = {}
            for _, index in self._tiers:
                for key, cid in index.items():
                    seen.setdefault(key, cid)
            self._emb_keys = list(seen.items())
            self._emb_matrix = self.embedder([self._display[k] for k, _ in self._emb_keys])
        query = self.embedder([text.strip().lower()])[0]
        sims = self._emb_matrix @ query
        best = int(np.argmax(sims))
        if float(sims[best]) < self.embedding_threshold:
            return None
        key, cid = self._emb_keys[best]
        return CategoryMatch(cid, self._names[cid], "embedding", float(sims[best]), key)


class HFTextEmbedder:
    """Embeddings de oraciones con `transformers` (mean pooling + normalización L2).

    El modelo se carga en el primer uso: si el léxico resuelve todo, no se descarga.
    """

    def __init__(self, model_id: str, device: str = "cpu", batch_size: int = 64) -> None:
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self.model = None

    def _load(self) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id).to(self.device).eval()

    def __call__(self, texts: list[str]) -> np.ndarray:
        import torch

        if self.model is None:
            self._load()
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = self.tokenizer(
                    texts[i : i + self.batch_size],
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.device)
                hidden = self.model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                emb = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                out.append(torch.nn.functional.normalize(emb, dim=1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 0), dtype=np.float32)


def load_lexicon(path: str | Path) -> dict[str, list[str]]:
    with Path(path).open(encoding="utf-8") as f:
        return {str(k): [str(t) for t in v] for k, v in (yaml.safe_load(f) or {}).items()}


def matcher_from_config(
    cfg: dict, categories: list[Category], device: str = "cpu"
) -> CategoryMatcher:
    """Matcher según `cfg['qa']['matcher']` (sinónimos LVIS vía `cfg['vocab']`)."""
    mcfg = cfg["qa"]["matcher"]
    lexicon = load_lexicon(mcfg["lexicon_es"]) if mcfg.get("lexicon_es") else {}
    synonyms = load_lvis_synonyms(cfg["vocab"], categories) if mcfg["lvis_synonyms"] else {}
    ecfg = mcfg["embeddings"]
    embedder = (
        HFTextEmbedder(ecfg["model_id"], device=device, batch_size=ecfg["batch_size"])
        if ecfg["enabled"]
        else None
    )
    return CategoryMatcher(
        categories,
        lexicon=lexicon,
        synonyms=synonyms,
        embedder=embedder,
        embedding_threshold=ecfg["threshold"],
    )
