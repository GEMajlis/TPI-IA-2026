import numpy as np
from PIL import Image

from eval.annotations import annotations_path, load_annotations
from scene.config import apply_dotlist
from scene.io.dataset import (
    image_record_from_path,
    list_images,
    load_image,
    load_questions,
    load_vocabulary,
)
from tests.conftest import DATA_ROOT, ROOT, requires_data


@requires_data
def test_vocabulario_120_categorias(base_cfg) -> None:
    vocab = load_vocabulary(base_cfg)
    assert len(vocab) == 120
    assert len({c.id for c in vocab}) == 120 and len({c.name for c in vocab}) == 120
    assert vocab[0].id == 1 and vocab[0].name == "sofa"


@requires_data
def test_splits(base_cfg) -> None:
    dev = list_images(base_cfg, "dev")
    test = list_images(base_cfg, "test")
    bench = list_images(base_cfg, "benchmark")
    assert (len(dev), len(test), len(bench)) == (120, 60, 20)
    assert dev[0].image_id == "tpi_0001" and test[0].image_id == "test_0001"
    assert {b.image_id for b in bench} <= {d.image_id for d in dev}
    assert len(load_questions(base_cfg)) == 60


@requires_data
def test_dimensiones_coinciden_con_la_metadata(base_cfg) -> None:
    # Las dimensiones se leen de la imagen; acá (solo en test) se contrastan con el GT.
    gt = load_annotations(annotations_path(base_cfg))
    for rec in list_images(base_cfg, "dev"):
        assert (rec.width, rec.height) == (gt[rec.image_id].width, gt[rec.image_id].height)


def test_funciona_sin_archivo_de_anotaciones(base_cfg, tmp_path) -> None:
    """La inferencia lista y carga imágenes aunque no exista annotations.json."""
    root = tmp_path / "data"
    images = root / "development" / "images"
    images.mkdir(parents=True)
    Image.fromarray(np.zeros((8, 6, 3), np.uint8)).save(images / "foo.jpg")
    cfg = apply_dotlist(base_cfg, [f"paths.data_root={root}"])
    (rec,) = list_images(cfg, "dev")
    assert (rec.image_id, rec.width, rec.height) == ("foo", 6, 8)
    assert load_image(rec).shape == (8, 6, 3)


def test_imagen_arbitraria(tmp_path) -> None:
    path = tmp_path / "externa.png"
    Image.fromarray(np.full((5, 9, 3), 255, np.uint8)).save(path)
    rec = image_record_from_path(path)
    assert (rec.image_id, rec.split, rec.width, rec.height) == ("externa", "external", 9, 5)


@requires_data
def test_gt_federado(base_cfg) -> None:
    gt = load_annotations(annotations_path(base_cfg))
    assert len(gt) == 120
    assert sum(len(g.instances) for g in gt.values()) == 1790
    for g in gt.values():
        # Una categoría negativa no puede tener anotaciones.
        assert not (g.neg_category_ids & g.positive_category_ids)
    img = gt["tpi_0001"]
    inst = img.instances[0]
    mask = img.mask(inst)
    assert mask.shape == (img.height, img.width) and mask.any()
    x0, y0, x1, y1 = inst.bbox_xyxy
    ys, xs = np.nonzero(mask)
    assert xs.min() >= int(x0) - 1 and xs.max() <= x1 + 1
    assert ys.min() >= int(y0) - 1 and ys.max() <= y1 + 1


def test_datos_de_competencia_versionados() -> None:
    # Los normativos versionados coinciden con la copia del dataset (si está).
    comp = ROOT / "data" / "competition"
    for name in ["scene_schema.json", "sample_submission.csv"]:
        assert (comp / name).is_file()
        if (DATA_ROOT / name).is_file():
            assert (comp / name).read_bytes() == (DATA_ROOT / name).read_bytes()
