"""Segmentación de instancias con SAM2 (vía `transformers`) usando las cajas detectadas como prompt.

El embedding de la imagen se calcula una sola vez y las cajas se procesan en lotes.
Con `multimask_output` SAM2 propone 3 máscaras por caja; queda la de mayor IoU predicho.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scene.detection.base import Detection
from scene.masks import rle
from scene.segmentation.postprocess import postprocess_mask


@dataclass
class SegmentedObject:
    detection: Detection
    mask: np.ndarray  # bool HxW, solo en memoria
    mask_score: float  # IoU predicho por SAM2
    raw_area: int = 0  # píxeles antes del post-proceso (para medir cuánto recorta)


def select_best_mask(masks: np.ndarray, iou_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """masks (N, K, H, W) e iou_scores (N, K) → la máscara de mayor IoU predicho por caja."""
    best = iou_scores.argmax(axis=1)
    idx = np.arange(len(best))
    return masks[idx, best], iou_scores[idx, best]


class Sam2Segmenter:
    """SAM2 con cajas como prompt."""

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        multimask_output: bool = True,
        box_batch_size: int = 32,
        mask_threshold: float = 0.0,
        postprocess: dict[str, Any] | None = None,
    ) -> None:
        import torch
        from transformers import Sam2Model, Sam2Processor

        self._torch = torch
        self.model_id = model_id
        self.device = device
        self.multimask_output = multimask_output
        self.box_batch_size = box_batch_size
        self.mask_threshold = mask_threshold
        self.pp_cfg = postprocess or {}
        self.processor = Sam2Processor.from_pretrained(model_id)
        self.model = Sam2Model.from_pretrained(model_id).to(device).eval()

    def params(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "multimask_output": self.multimask_output,
            "box_batch_size": self.box_batch_size,
            "mask_threshold": self.mask_threshold,
            "postprocess": self.pp_cfg,
        }

    def segment(self, image: np.ndarray, dets: list[Detection]) -> list[SegmentedObject]:
        if not dets:
            return []
        torch = self._torch
        inputs = self.processor(images=image, return_tensors="pt")
        original_sizes = inputs["original_sizes"]
        with torch.inference_mode():
            embeddings = self.model.get_image_embeddings(inputs["pixel_values"].to(self.device))
        out: list[SegmentedObject] = []
        for start in range(0, len(dets), self.box_batch_size):
            chunk = dets[start : start + self.box_batch_size]
            boxes = self.processor(
                input_boxes=[[list(d.box_xyxy) for d in chunk]],
                original_sizes=original_sizes,
                return_tensors="pt",
            )["input_boxes"]
            with torch.inference_mode():
                pred = self.model(
                    image_embeddings=embeddings,
                    input_boxes=boxes.to(self.device),
                    multimask_output=self.multimask_output,
                )
            masks = self.processor.post_process_masks(
                pred.pred_masks.float().cpu(), original_sizes, mask_threshold=self.mask_threshold
            )[0].numpy()
            best, scores = select_best_mask(masks, pred.iou_scores[0].float().cpu().numpy())
            for d, m, s in zip(chunk, best, scores, strict=True):
                m = m.astype(bool)
                out.append(
                    SegmentedObject(
                        detection=d,
                        mask=postprocess_mask(m, d.box_xyxy, self.pp_cfg),
                        mask_score=float(s),
                        raw_area=int(m.sum()),
                    )
                )
        return out


def build_segmenter(cfg: dict, device: str) -> Sam2Segmenter:
    s = cfg["segmentation"]
    return Sam2Segmenter(
        model_id=s["model_id"],
        device=device,
        multimask_output=s["multimask_output"],
        box_batch_size=s["box_batch_size"],
        mask_threshold=s["mask_threshold"],
        postprocess=s["postprocess"],
    )


def save_segmented(objs: list[SegmentedObject], path: str | Path) -> None:
    """Caché por imagen: detección + RLE + score (no se guardan matrices completas)."""
    payload = [
        {
            "detection": o.detection.to_dict(),
            "mask": rle.encode(o.mask),
            "mask_score": o.mask_score,
            "raw_area": o.raw_area,
        }
        for o in objs
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def load_segmented(path: str | Path) -> list[SegmentedObject]:
    return [
        SegmentedObject(
            detection=Detection.from_dict(o["detection"]),
            mask=rle.decode(o["mask"]),
            mask_score=o["mask_score"],
            raw_area=o["raw_area"],
        )
        for o in json.loads(Path(path).read_text(encoding="utf-8"))
    ]
