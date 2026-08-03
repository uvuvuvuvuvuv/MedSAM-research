#!/usr/bin/env python3

import json
import re
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np


ROOT = Path("/storage/baiyuting/data/MedSAM-main/data/organized")
SYN_ROOT = ROOT / "synapse"
BTCV_ROOT = ROOT / "btcv"

OUTPUT = Path(
    "/storage/baiyuting/data/MedSAM-main/data/processed/"
    "synapse/fold_0/meta/synapse_spacing_reference_audit.json"
)


def case_number(name: str) -> int:
    values = re.findall(r"\d+", name)
    if not values:
        raise ValueError(f"Cannot parse numeric ID: {name}")
    return int(values[-1])


def find_h5_image_dataset(f: h5py.File):
    for key in ["image", "img", "imgs", "data"]:
        if key in f:
            return key, f[key]

    for key in f.keys():
        obj = f[key]
        if isinstance(obj, h5py.Dataset) and obj.ndim == 3:
            return key, obj

    raise ValueError(f"Cannot find 3D image dataset; keys={list(f.keys())}")


def read_h5_spacing(f: h5py.File):
    for key in ["spacing", "spacings", "pixdim"]:
        if key in f:
            arr = np.asarray(f[key]).reshape(-1)
            if len(arr) >= 3:
                return [float(arr[0]), float(arr[1]), float(arr[2])]

    for key in ["spacing", "spacings", "pixdim"]:
        if key in f.attrs:
            arr = np.asarray(f.attrs[key]).reshape(-1)
            if len(arr) >= 3:
                return [float(arr[0]), float(arr[1]), float(arr[2])]

    return None


syn_case_dirs = sorted(
    list((SYN_ROOT / "train" / "cases").glob("case*"))
    + list((SYN_ROOT / "test" / "cases").glob("case*"))
)

records = {}
errors = []

for case_dir in syn_case_dirs:
    case_id = case_dir.name
    number = case_number(case_id)
    btcv_id = f"img{number:04d}"

    btcv_candidates = sorted(BTCV_ROOT.rglob(f"{btcv_id}/image.nii.gz"))

    if len(btcv_candidates) != 1:
        errors.append({
            "case_id": case_id,
            "reason": "btcv_reference_not_unique",
            "candidates": [str(x) for x in btcv_candidates],
        })
        continue

    btcv_path = btcv_candidates[0]
    nii = nib.load(str(btcv_path))

    btcv_shape = [int(x) for x in nii.shape[:3]]
    btcv_spacing = [float(x) for x in nii.header.get_zooms()[:3]]
    orientation = list(nib.aff2axcodes(nii.affine))

    npz_dir = case_dir / "aux" / "slices"
    h5_path = case_dir / "image.h5"

    source_format = None
    synapse_depth = None
    h5_spacing = None
    source_detail = None

    if npz_dir.exists():
        npz_files = sorted(npz_dir.glob("*.npz"))
        source_format = "npz_slices"
        synapse_depth = len(npz_files)
        source_detail = str(npz_dir)

    elif h5_path.exists():
        source_format = "h5_volume"
        source_detail = str(h5_path)

        with h5py.File(h5_path, "r") as f:
            image_key, image_ds = find_h5_image_dataset(f)
            synapse_depth = int(image_ds.shape[0])
            h5_spacing = read_h5_spacing(f)
            source_detail += f"::{image_key}"

    else:
        errors.append({
            "case_id": case_id,
            "reason": "synapse_source_not_found",
            "case_dir": str(case_dir),
        })
        continue

    depth_match = synapse_depth == btcv_shape[-1]

    if not depth_match:
        errors.append({
            "case_id": case_id,
            "reason": "depth_mismatch",
            "synapse_depth": synapse_depth,
            "btcv_shape": btcv_shape,
            "btcv_path": str(btcv_path),
        })

    records[case_id] = {
        "case_id": case_id,
        "mapped_btcv_case_id": btcv_id,
        "source_format": source_format,
        "synapse_source": source_detail,
        "synapse_num_slices": synapse_depth,
        "btcv_reference_path": str(btcv_path),
        "btcv_shape_xyz": btcv_shape,
        "spacing_xyz": btcv_spacing,
        "orientation": orientation,
        "h5_embedded_spacing": h5_spacing,
        "depth_match": depth_match,
        "spacing_source": "corresponding_btcv_nifti_header",
    }

report = {
    "dataset": "synapse",
    "num_synapse_cases": len(syn_case_dirs),
    "num_mapped_cases": len(records),
    "num_depth_matches": sum(
        int(x["depth_match"]) for x in records.values()
    ),
    "num_errors": len(errors),
    "cases": records,
    "errors": errors,
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(
    json.dumps(report, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("output:", OUTPUT)
print("num_synapse_cases:", report["num_synapse_cases"])
print("num_mapped_cases:", report["num_mapped_cases"])
print("num_depth_matches:", report["num_depth_matches"])
print("num_errors:", report["num_errors"])

if errors:
    print("\nErrors:")
    for error in errors:
        print(error)

print("\nSpacing examples:")
for case_id in sorted(records)[:5]:
    rec = records[case_id]
    print(
        case_id,
        "->",
        rec["mapped_btcv_case_id"],
        "depth=", rec["synapse_num_slices"],
        "spacing=", rec["spacing_xyz"],
        "h5_spacing=", rec["h5_embedded_spacing"],
    )
