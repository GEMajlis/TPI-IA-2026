"""Detector OWLv2 (línea base) vía `transformers`.

OWLv2 rellena la imagen hasta hacerla cuadrada (abajo y a la derecha) y la reescala a 960 px.
Sus cajas salen normalizadas respecto de ese cuadrado, así que para volver a píxeles de la imagen
original se multiplica por el lado mayor (no por ancho y alto por separado).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from scene.detection.base import Detection, Detector
from scene.detection.vocabulary import Vocabulary


def owl_boxes_to_xyxy(boxes_cxcywh: np.ndarray, width: int, height: int) -> np.ndarray:
    """Cajas (cx, cy, w, h) normalizadas al cuadrado con padding → xyxy en píxeles originales."""
    side = max(width, height)
    cx, cy, w, h = (boxes_cxcywh[:, i] * side for i in range(4))
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, width)
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, height)
    return xyxy


def decode_detections(
    scores: np.ndarray,
    boxes_xyxy: np.ndarray,
    vocab: Vocabulary,
    score_floor: float,
    raw_topk: int,
    labels_per_box: int,
    source: str = "owlv2",
) -> list[Detection]:
    """Scores por (caja, query) → detecciones crudas.

    1. Por cada caja y categoría queda la query (nombre o sinónimo) con mayor score.
    2. Por cada caja se conservan las `labels_per_box` categorías con mayor score.
    3. Se descarta lo que está bajo `score_floor` y se ordena; quedan como máximo `raw_topk`.
    """
    by_cat = vocab.by_category()
    cat_ids = [cid for cid, qs in by_cat.items() if qs]
    query_index = {id(q): i for i, q in enumerate(vocab.queries)}
    n_boxes = scores.shape[0]
    cat_scores = np.empty((n_boxes, len(cat_ids)), dtype=np.float32)
    best_query = np.empty((n_boxes, len(cat_ids)), dtype=np.int64)
    for j, cid in enumerate(cat_ids):
        qidx = np.array([query_index[id(q)] for q in by_cat[cid]])
        sub = scores[:, qidx]
        cat_scores[:, j] = sub.max(axis=1)
        best_query[:, j] = qidx[sub.argmax(axis=1)]

    k = min(labels_per_box, len(cat_ids))
    top = np.argsort(-cat_scores, axis=1, kind="stable")[:, :k]
    rows = np.repeat(np.arange(n_boxes), k)
    cols = top.reshape(-1)
    sel_scores = cat_scores[rows, cols]
    valid = (sel_scores >= score_floor) & (boxes_xyxy[rows, 2] > boxes_xyxy[rows, 0])
    valid &= boxes_xyxy[rows, 3] > boxes_xyxy[rows, 1]
    rows, cols, sel_scores = rows[valid], cols[valid], sel_scores[valid]
    order = np.argsort(-sel_scores, kind="stable")[:raw_topk]

    names = {c.id: c.name for c in vocab.categories}
    dets: list[Detection] = []
    for r, c in zip(rows[order], cols[order], strict=True):
        q = vocab.queries[best_query[r, c]]
        dets.append(
            Detection(
                box_xyxy=tuple(float(v) for v in boxes_xyxy[r]),
                score=float(cat_scores[r, c]),
                category_id=q.category_id,
                category_name=names[q.category_id],
                source=source,
                raw_query=q.text,
            )
        )
    return dets


class Owlv2Detector(Detector):
    """OWLv2 con el vocabulario fijo. Devuelve detecciones crudas (el NMS va en postprocess)."""

    source = "owlv2"

    def __init__(
        self,
        vocab: Vocabulary,
        model_id: str,
        device: str = "cpu",
        score_floor: float = 0.01,
        raw_topk: int = 300,
        labels_per_box: int = 1,
        query_batch_size: int | None = None,
    ) -> None:
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self._torch = torch
        self.vocab = vocab
        self.model_id = model_id
        self.device = device
        self.score_floor = score_floor
        self.raw_topk = raw_topk
        self.labels_per_box = labels_per_box
        self.query_batch_size = query_batch_size
        self.processor = Owlv2Processor.from_pretrained(model_id)
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model = Owlv2ForObjectDetection.from_pretrained(model_id, torch_dtype=dtype)
        self.model.to(device).eval()
        self.dtype = dtype

    def params(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "n_queries": len(self.vocab.queries),
            "template": self.vocab.template,
            "score_floor": self.score_floor,
            "raw_topk": self.raw_topk,
            "labels_per_box": self.labels_per_box,
            "query_batch_size": self.query_batch_size,
        }

    def _scores_and_boxes(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Corre el modelo: scores (cajas × queries) y cajas cxcywh normalizadas."""
        torch = self._torch
        prompts = self.vocab.prompts()
        step = self.query_batch_size or len(prompts)
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.to(self.device, self.dtype)
        chunks, boxes = [], None
        for start in range(0, len(prompts), step):
            text = self.processor(text=[prompts[start : start + step]], return_tensors="pt")
            with torch.inference_mode():
                out = self.model(
                    input_ids=text["input_ids"].to(self.device),
                    attention_mask=text["attention_mask"].to(self.device),
                    pixel_values=pixel_values,
                )
            chunks.append(out.logits[0].sigmoid().float().cpu().numpy())
            boxes = out.pred_boxes[0].float().cpu().numpy()
        return np.concatenate(chunks, axis=1), boxes

    def detect(self, image: np.ndarray) -> list[Detection]:
        height, width = image.shape[:2]
        scores, boxes = self._scores_and_boxes(image)
        return decode_detections(
            scores,
            owl_boxes_to_xyxy(boxes, width, height),
            self.vocab,
            self.score_floor,
            self.raw_topk,
            self.labels_per_box,
            self.source,
        )


def build_detector(cfg: dict, vocab: Vocabulary, device: str) -> Detector:
    """Detector según `cfg['detection']['detector']`."""
    det_cfg = cfg["detection"]
    name = det_cfg["detector"]
    if name == "owlv2":
        o = det_cfg["owlv2"]
        return Owlv2Detector(
            vocab,
            model_id=o["model_id"],
            device=device,
            score_floor=o["score_floor"],
            raw_topk=o["raw_topk"],
            labels_per_box=o["labels_per_box"],
            query_batch_size=o.get("query_batch_size"),
        )
    raise ValueError(f"detection.detector desconocido: {name!r}")
