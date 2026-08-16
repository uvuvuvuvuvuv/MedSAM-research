from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


IGNORE_LABEL = 255


def sigmoid_np(logits: np.ndarray) -> np.ndarray:
    """
    Stable sigmoid for MedSAM pixel logits.
    """
    logits = np.asarray(logits, dtype=np.float32)
    return 1.0 / (
        1.0
        + np.exp(
            -np.clip(logits, -40.0, 40.0)
        )
    )


def aggregate_same_class_evidence(
    class_probabilities: Dict[int, np.ndarray],
    class_support_masks: Dict[int, np.ndarray],
    label_id: int,
    probability: np.ndarray,
    support_mask: np.ndarray,
) -> None:
    """
    Aggregate multiple instances of the SAME semantic class.

    support_mask:
        Final selected/refined MedSAM best_mask for this instance.

    probability:
        Pixel-level foreground probability belonging to the SAME
        multimask candidate as support_mask.

    Same-class overlaps are not conflicts:
        support = logical OR
        probability = pixel-wise MAX
    """
    label_id = int(label_id)

    if label_id <= 0 or label_id == IGNORE_LABEL:
        raise ValueError(
            f"Illegal foreground label_id={label_id}"
        )

    probability = np.asarray(
        probability,
        dtype=np.float32,
    )
    support_mask = np.asarray(
        support_mask,
        dtype=bool,
    )

    if probability.shape != support_mask.shape:
        raise ValueError(
            "Probability/support shape mismatch: "
            f"{probability.shape} vs {support_mask.shape}"
        )

    probability = np.nan_to_num(
        probability,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    probability = np.clip(
        probability,
        0.0,
        1.0,
    )

    # Probability is only meaningful for class arbitration
    # where the selected proposal actually supports foreground.
    supported_probability = np.where(
        support_mask,
        probability,
        0.0,
    ).astype(np.float32)

    if label_id not in class_probabilities:
        class_probabilities[label_id] = (
            supported_probability.copy()
        )
        class_support_masks[label_id] = (
            support_mask.copy()
        )
        return

    if (
        class_probabilities[label_id].shape
        != probability.shape
    ):
        raise ValueError(
            "Same-class probability shape mismatch for "
            f"label {label_id}"
        )

    if (
        class_support_masks[label_id].shape
        != support_mask.shape
    ):
        raise ValueError(
            "Same-class support shape mismatch for "
            f"label {label_id}"
        )

    np.maximum(
        class_probabilities[label_id],
        supported_probability,
        out=class_probabilities[label_id],
    )

    class_support_masks[label_id] |= support_mask


def resolve_multiclass_evidence(
    class_probabilities: Dict[int, np.ndarray],
    class_support_masks: Dict[int, np.ndarray],
    union_all: np.ndarray,
    tie_epsilon: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    V2 probability-aware multiclass resolver.

    IMPORTANT
    ---------
    Existing V1 foreground support is preserved.

    Pixel semantics:

      outside all canonical boxes
          -> 0

      inside box union, but no selected MedSAM proposal
      confirms foreground
          -> 255

      exactly one semantic class confirms foreground
          -> that class

      >=2 different semantic classes confirm foreground
          -> choose the class with the highest pixel probability

      numerical top-1/top-2 probability tie
          -> 255

    No additional confidence-margin hyperparameter is introduced.
    Pixel probabilities are used ONLY to resolve true cross-class
    overlaps between already-confirmed best masks.
    """
    union_all = np.asarray(
        union_all,
        dtype=bool,
    )

    confirmed_fg = np.zeros(
        union_all.shape,
        dtype=np.uint8,
    )

    tri = np.zeros(
        union_all.shape,
        dtype=np.uint8,
    )
    tri[union_all] = IGNORE_LABEL

    if not class_support_masks:
        stats = {
            "num_box_pixels": int(
                union_all.sum()
            ),
            "num_confirmed_fg_pixels": 0,
            "num_unknown_pixels": int(
                union_all.sum()
            ),
            "num_single_class_pixels": 0,
            "num_multiclass_competition_pixels": 0,
            "num_resolved_competition_pixels": 0,
            "num_tied_competition_pixels": 0,
        }
        return confirmed_fg, tri, stats

    labels = sorted(
        int(x)
        for x in class_support_masks.keys()
    )

    if set(labels) != set(
        int(x)
        for x in class_probabilities.keys()
    ):
        raise ValueError(
            "class_probabilities and class_support_masks "
            "must contain identical label IDs"
        )

    probability_stack = np.stack(
        [
            np.asarray(
                class_probabilities[label],
                dtype=np.float32,
            )
            for label in labels
        ],
        axis=0,
    )

    support_stack = np.stack(
        [
            np.asarray(
                class_support_masks[label],
                dtype=bool,
            )
            for label in labels
        ],
        axis=0,
    )

    if probability_stack.shape[1:] != union_all.shape:
        raise ValueError(
            "Probability map shape mismatch: "
            f"{probability_stack.shape[1:]} "
            f"vs union {union_all.shape}"
        )

    if support_stack.shape[1:] != union_all.shape:
        raise ValueError(
            "Support map shape mismatch: "
            f"{support_stack.shape[1:]} "
            f"vs union {union_all.shape}"
        )

    # Arbitration is only valid inside canonical box union.
    support_stack &= union_all[None, ...]

    active_count = support_stack.sum(
        axis=0
    )

    # Unsupported classes must never win an argmax.
    competition_scores = np.where(
        support_stack,
        probability_stack,
        -np.inf,
    )

    best_index = np.argmax(
        competition_scores,
        axis=0,
    )

    best_probability = np.take_along_axis(
        competition_scores,
        best_index[None, ...],
        axis=0,
    )[0]

    label_lut = np.asarray(
        labels,
        dtype=np.uint8,
    )
    best_label = label_lut[
        best_index
    ]

    single_class = (
        union_all
        & (active_count == 1)
    )

    multiclass_competition = (
        union_all
        & (active_count >= 2)
    )

    if len(labels) >= 2:
        sorted_scores = np.sort(
            competition_scores,
            axis=0,
        )
        second_probability = (
            sorted_scores[-2]
        )
    else:
        second_probability = np.full(
            union_all.shape,
            -np.inf,
            dtype=np.float32,
        )

    # Compute the top-1/top-2 gap ONLY on valid multiclass
    # competition pixels. Outside those pixels competition_scores
    # may contain -inf, and evaluating (-inf) - (-inf) would emit
    # an unnecessary NumPy RuntimeWarning.
    valid_competition = (
        multiclass_competition
        & np.isfinite(best_probability)
        & np.isfinite(second_probability)
    )

    probability_gap = np.full(
        union_all.shape,
        np.inf,
        dtype=np.float32,
    )

    probability_gap[valid_competition] = np.abs(
        best_probability[valid_competition]
        - second_probability[valid_competition]
    )

    tie = (
        valid_competition
        & (
            probability_gap
            <= float(tie_epsilon)
        )
    )

    resolved_competition = (
        multiclass_competition
        & (~tie)
    )

    accepted = (
        single_class
        | resolved_competition
    )

    confirmed_fg[accepted] = (
        best_label[accepted]
    )
    tri[accepted] = (
        best_label[accepted]
    )

    # tie remains 255 because tri was initialized as
    # 255 throughout union_all.

    stats = {
        "num_box_pixels": int(
            union_all.sum()
        ),
        "num_confirmed_fg_pixels": int(
            accepted.sum()
        ),
        "num_unknown_pixels": int(
            (
                union_all
                & (tri == IGNORE_LABEL)
            ).sum()
        ),
        "num_single_class_pixels": int(
            single_class.sum()
        ),
        "num_multiclass_competition_pixels": int(
            multiclass_competition.sum()
        ),
        "num_resolved_competition_pixels": int(
            resolved_competition.sum()
        ),
        "num_tied_competition_pixels": int(
            tie.sum()
        ),
    }

    return confirmed_fg, tri, stats
