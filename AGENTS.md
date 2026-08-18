# MedSAM Research Development Rules

## Repository Role

This repository contains the MedSAM-side research pipeline, including:

- teacher inference
- annotation/sample selection
- MedSAM adaptation
- pseudo-label generation
- experiment protocol utilities

The downstream Student implementation is maintained separately in
`Swin-UMamba-research`.

## Git Policy

- Never develop directly on `main`.
- Never develop directly on `clean-paper-v1`.
- New research development must use `exp/*` branches.
- Do not rewrite historical commits.
- Do not move or recreate formal/release tags.
- Always inspect `git status` before modifying files.

## Frozen Research Assets

Do not modify:

- `/storage/baiyuting/data/MedSAM-main`
- frozen formal experiment outputs
- final-lock directories
- frozen 3D protocols

Do not rerun frozen formal experiments unless explicitly requested.

## Development Policy

New experimental ideas should be implemented as new methods, configs,
or protocol variants rather than silently modifying existing formal methods.

Preserve compatibility with previous experiment outputs whenever possible.

Use explicit experiment/method names for new protocols.

## Validation

Before proposing a commit:

1. Show `git status`.
2. Show `git diff --stat`.
3. Run appropriate syntax or smoke tests.
4. Summarize exactly what changed.
5. Explain whether the modification changes the experimental protocol.

Do not commit or push unless explicitly requested.