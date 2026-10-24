"""Vocabulario de detección: nombres limpios, sinónimos LVIS opcionales y mapeo query → categoría.

Solo usa el vocabulario de categorías (nombres y sinónimos públicos de LVIS), nunca anotaciones.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from scene.io.dataset import Category

VOCAB_MODES = ("clean", "synonyms")


def clean_name(name: str) -> str:
    """`glass_(drink_container)` → `glass`: sin desambiguación, `_` → espacio, minúsculas."""
    name = re.sub(r"\([^)]*\)", " ", name.replace("_", " "))
    return " ".join(name.lower().split())


@dataclass(frozen=True)
class Query:
    """Texto que se le pasa al detector y la categoría a la que pertenece."""

    text: str
    category_id: int
    category_name: str
    is_synonym: bool = False


@dataclass
class Vocabulary:
    """Lista ordenada de queries. El índice de la query es el que devuelve el detector."""

    categories: list[Category]
    queries: list[Query]
    template: str = "{}"
    # Sinónimos descartados por colisionar entre categorías (para trazabilidad).
    dropped: dict[str, list[int]] = field(default_factory=dict)

    def prompts(self) -> list[str]:
        """Textos finales para el detector (query dentro del template)."""
        return [self.template.format(q.text) for q in self.queries]

    def query_to_category(self) -> list[int]:
        """category_id de cada query, en el mismo orden que `prompts()`."""
        return [q.category_id for q in self.queries]

    def by_category(self) -> dict[int, list[Query]]:
        out: dict[int, list[Query]] = {c.id: [] for c in self.categories}
        for q in self.queries:
            out[q.category_id].append(q)
        return out


def build_vocabulary(
    categories: list[Category],
    mode: str = "clean",
    synonyms: dict[int, list[str]] | None = None,
    template: str = "{}",
    max_synonyms: int | None = None,
) -> Vocabulary:
    """Arma las queries. En modo `synonyms` agrega sinónimos sin repetir textos entre categorías.

    Si un sinónimo coincide con el nombre de otra categoría o con un sinónimo de otra, se descarta
    en todas (así cada query mapea a una sola categoría).
    """
    if mode not in VOCAB_MODES:
        raise ValueError(f"vocab.mode inválido {mode!r}; opciones: {VOCAB_MODES}")
    main_names = {c.id: clean_name(c.name) for c in categories}
    if len(set(main_names.values())) != len(main_names):
        raise ValueError("Hay categorías con el mismo nombre limpio")

    extra: dict[int, list[str]] = {c.id: [] for c in categories}
    if mode == "synonyms":
        for cid in extra:
            seen = {main_names[cid]}
            for syn in (synonyms or {}).get(cid, []):
                text = clean_name(syn)
                if text and text not in seen:
                    seen.add(text)
                    extra[cid].append(text)
            if max_synonyms is not None:
                extra[cid] = extra[cid][:max_synonyms]

    # Quién usa cada texto: si lo usa más de una categoría, se descarta como sinónimo.
    owners: dict[str, set[int]] = {}
    for cid, name in main_names.items():
        owners.setdefault(name, set()).add(cid)
    for cid, texts in extra.items():
        for text in texts:
            owners.setdefault(text, set()).add(cid)
    dropped = {t: sorted(ids) for t, ids in owners.items() if len(ids) > 1}

    queries: list[Query] = []
    for cat in categories:
        queries.append(Query(main_names[cat.id], cat.id, cat.name))
        for text in extra[cat.id]:
            if text not in dropped:
                queries.append(Query(text, cat.id, cat.name, is_synonym=True))
    return Vocabulary(categories=categories, queries=queries, template=template, dropped=dropped)


def parse_lvis_categories(source: str) -> list[dict]:
    """Extrae `LVIS_CATEGORIES` de `lvis_v1_categories.py` (detectron2) sin ejecutar código."""
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "LVIS_CATEGORIES" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError("No se encontró LVIS_CATEGORIES en la fuente")


def match_lvis_synonyms(
    categories: list[Category], lvis_categories: list[dict], aliases: dict[str, str] | None = None
) -> dict[int, list[str]]:
    """Sinónimos LVIS de cada categoría de la competencia.

    Busca por: alias explícito (nombre LVIS crudo) → nombre LVIS limpio → algún sinónimo limpio.
    """
    aliases = aliases or {}
    by_raw = {c["name"]: c for c in lvis_categories}
    by_clean = {clean_name(c["name"]): c for c in lvis_categories}
    out: dict[int, list[str]] = {}
    missing: list[str] = []
    for cat in categories:
        name = clean_name(cat.name)
        entry = by_raw.get(aliases[name]) if name in aliases else by_clean.get(name)
        if entry is None:
            entry = next(
                (c for c in lvis_categories if name in {clean_name(s) for s in c["synonyms"]}),
                None,
            )
        if entry is None:
            missing.append(cat.name)
            continue
        out[cat.id] = list(entry["synonyms"])
    if missing:
        raise ValueError(f"Categorías sin entrada LVIS (agregar en vocab.lvis_aliases): {missing}")
    return out


def load_lvis_synonyms(vocab_cfg: dict, categories: list[Category]) -> dict[int, list[str]]:
    """Descarga (una vez) los sinónimos LVIS públicos y los cachea en JSON."""
    cache = Path(vocab_cfg["synonyms_cache"])
    if cache.is_file():
        data = json.loads(cache.read_text(encoding="utf-8"))
        return {int(k): v for k, v in data["synonyms"].items()}
    url = vocab_cfg["lvis_categories_url"]
    with urllib.request.urlopen(url, timeout=60) as resp:
        source = resp.read().decode("utf-8")
    synonyms = match_lvis_synonyms(
        categories, parse_lvis_categories(source), vocab_cfg.get("lvis_aliases")
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_url": url,
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "synonyms": {str(k): v for k, v in synonyms.items()},
    }
    cache.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    return synonyms


def vocabulary_from_config(cfg: dict, categories: list[Category]) -> Vocabulary:
    """Vocabulario según `cfg['vocab']` (descarga sinónimos solo en modo `synonyms`)."""
    vcfg = cfg["vocab"]
    synonyms = load_lvis_synonyms(vcfg, categories) if vcfg["mode"] == "synonyms" else None
    return build_vocabulary(
        categories,
        mode=vcfg["mode"],
        synonyms=synonyms,
        template=vcfg["prompt_template"],
        max_synonyms=vcfg.get("max_synonyms"),
    )
