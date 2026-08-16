# Baseline

This directory documents the baseline pseudo-label generation pipeline.

## Baseline definition

The baseline uses the original frozen MedSAM teacher with Box prompts to
generate tri-state pseudo-labels.

Canonical implementation:

    ../../generate_pseudo_labels.py

Shared probability-aware multi-class arbitration:

    ../common/multiclass_resolver.py

No Teacher adaptation or Full-label acquisition is used in the baseline.

## Relation to Idea1

Idea1 extends this baseline with:

1. a small Full-label acquisition budget;
2. hard-sample selection;
3. MedSAM mask-decoder adaptation;
4. regenerated Box pseudo-labels.

The baseline and Idea1 therefore share the same underlying pseudo-label
generation logic whenever applicable.
