"""Shared instance-level GT matching utilities.

Used by both 02_build_support_template and 03_train_medsam_full_only to avoid
duplicated connected-component / bbox-IoU matching logic.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Bbox helpers
# ---------------------------------------------------------------------------


def bbox_iou(a: list[float], b: list[float]) -> float:
    """Intersection-over-Union of two axis-aligned bounding boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x1 >= x2 or y1 >= y2:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = max((a[2] - a[0]) * (a[3] - a[1]), 0)
    area_b = max((b[2] - b[0]) * (b[3] - b[1]), 0)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def bbox_from_binary(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return the tight bounding box *(x1, y1, x2, y2)* of a binary mask,
    or *None* if the mask is empty."""
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None
    return (
        int(cols.min()),
        int(rows.min()),
        int(cols.max()) + 1,
        int(rows.max()) + 1,
    )


# ---------------------------------------------------------------------------
# GT loading
# ---------------------------------------------------------------------------


def load_gt_array(path: Path) -> np.ndarray:
    """Load a ground-truth mask from *.npy*, returning int64 (H, W)."""
    gt = np.load(str(path))
    if gt.ndim == 3:
        if gt.shape[0] == 1:
            gt = gt.squeeze(0)
        elif gt.shape[-1] == 1:
            gt = gt.squeeze(-1)
        else:
            raise ValueError(f"Unexpected GT shape {gt.shape} from {path}")
    return gt.astype(np.int64)


# ---------------------------------------------------------------------------
# Instance matching
# ---------------------------------------------------------------------------


def match_instance_mask(
    gt: np.ndarray,
    label_id: int,
    bbox: list[float],
    component_id: int | None,
) -> np.ndarray:
    """Return a boolean mask *(H, W)* for the matched instance.

    *component_id* takes priority (1-based OpenCV label).
    When *component_id* is *None*, the connected component whose tight
    bbox has the highest IoU with *bbox* is selected.

    Raises *ValueError* when no valid target is found.
    """
    mask_all = gt == int(label_id)

    if not mask_all.any():
        raise ValueError(
            f"label_id={label_id} not found in GT"
        )

    mask_all_uint8 = mask_all.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_all_uint8, connectivity=8
    )

    if num_labels <= 1:
        raise ValueError(
            f"No connected components found for label_id={label_id}"
        )

    if component_id is not None:
        if not isinstance(component_id, int) or component_id < 1:
            raise ValueError(
                f"component_id must be positive int, got {component_id!r} "
                f"for label_id={label_id}"
            )
        if component_id >= num_labels:
            raise ValueError(
                f"component_id={component_id} out of range "
                f"(found {num_labels - 1} components) for label_id={label_id}"
            )
        selected_mask = labels == int(component_id)
        if not selected_mask.any():
            raise ValueError(
                f"component_id={component_id} produced empty mask "
                f"for label_id={label_id}"
            )
    else:
        best_idx: int | None = None
        best_iou = -1.0
        for comp_idx in range(1, num_labels):
            comp_mask = labels == comp_idx
            rows, cols = np.where(comp_mask)
            if len(rows) == 0:
                continue
            comp_bbox = [
                float(cols.min()),
                float(rows.min()),
                float(cols.max()) + 1,
                float(rows.max()) + 1,
            ]
            iou = bbox_iou(bbox, comp_bbox)
            if iou > best_iou:
                best_iou = iou
                best_idx = comp_idx

        if best_idx is None or best_iou <= 0.0:
            raise ValueError(
                f"No connected component overlaps with bbox={bbox} "
                f"for label_id={label_id}"
            )
        selected_mask = labels == best_idx
        if not selected_mask.any():
            raise ValueError(
                f"bbox-matched component produced empty mask "
                f"for label_id={label_id}"
            )

    return selected_mask


def load_instance_target(
    path: Path,
    device: torch.device,
    label_id: int,
    bbox: list[float],
    component_id: int | None,
) -> torch.Tensor:
    """Load a GT *.npy* file and return a *(1, 1, H, W)* torch float tensor
    for the matched instance.

    Convenience wrapper used by 03_train_medsam_full_only.
    """
    gt = load_gt_array(path)
    mask = match_instance_mask(gt, label_id, bbox, component_id)
    target = mask.astype(np.float32)
    return torch.from_numpy(target).unsqueeze(0).unsqueeze(0).float().to(device)
