"""Arma `submission.csv` a partir de los SceneJSON de uno o más runs y lo valida completo.

Uso:
    python scripts/make_submission.py --run outputs/runs/<run_id>
    python scripts/make_submission.py --config configs/base.yaml configs/kaggle.yaml \\
        --run latest --out /kaggle/working/submission.csv

Si una imagen está en varios runs, gana el último de la lista. Falla si falta alguna de las 80
filas o si algún SceneJSON no valida (no escribe un submission inválido).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scene.config import load_config
from scene.io.submission import build_submission, read_submission, write_submission
from scene.pipeline import load_scenes


def resolve_run(cfg: dict, run: str) -> Path:
    if run != "latest":
        return Path(run)
    runs = sorted(p for p in Path(cfg["paths"]["outputs"]).iterdir() if (p / "scenes").is_dir())
    if not runs:
        raise SystemExit("No hay runs con scenes/: correr antes scripts/run_pipeline.py")
    return runs[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", nargs="+", default=["configs/base.yaml"])
    parser.add_argument("--run", nargs="+", required=True, help="runs con scenes/ o 'latest'")
    parser.add_argument("--out", help="ruta del CSV (por defecto, dentro del último run)")
    args = parser.parse_args()

    cfg = load_config(*args.config)
    runs = [resolve_run(cfg, r) for r in args.run]
    scenes: dict = {}
    for run in runs:
        scenes.update(load_scenes(run))
    rows, errors = build_submission(scenes, cfg)
    if errors:
        for iid, errs in list(errors.items())[:10]:
            print(f"{iid}: {errs[:3]}")
        raise SystemExit(f"{len(errors)} filas con error: no se escribe el submission")

    out = Path(args.out) if args.out else runs[-1] / cfg["paths"]["submission"]
    write_submission(rows, out)
    back = read_submission(out)  # chequeo de ida y vuelta del CSV
    assert list(back) == [iid for iid, _ in rows]
    n_obj = sum(len(s["objects"]) for s in back.values())
    n_rel = sum(len(s["relations"]) for s in back.values())
    size_mb = out.stat().st_size / 2**20
    print(f"{out}: {len(rows)} filas, {n_obj} objetos, {n_rel} relaciones, {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
