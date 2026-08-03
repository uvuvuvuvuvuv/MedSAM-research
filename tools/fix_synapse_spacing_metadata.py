import json
import os
import re
import shutil
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List


FOLD_ROOT = (
    "/storage/baiyuting/data/MedSAM-main/"
    "data/processed/synapse/fold_0"
)

AUDIT_PATH = os.path.join(
    FOLD_ROOT,
    "meta",
    "synapse_spacing_reference_audit.json",
)

MANIFEST_PATH = os.path.join(
    FOLD_ROOT,
    "meta",
    "manifest.json",
)

GEOMETRY_PATH = os.path.join(
    FOLD_ROOT,
    "meta",
    "geometry_meta.json",
)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def atomic_save_json(obj: Any, path: str):
    tmp_path = path + ".tmp_spacing_fix"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp_path, path)


def get_manifest_records(manifest: Any) -> List[Dict[str, Any]]:
    if isinstance(manifest, list):
        return manifest

    if isinstance(manifest, dict):
        if isinstance(manifest.get("records"), list):
            return manifest["records"]

        if isinstance(manifest.get("items"), list):
            return manifest["items"]

        records = [
            value
            for value in manifest.values()
            if isinstance(value, dict) and "case_id" in value
        ]
        if records:
            return records

    raise TypeError(
        f"Unsupported manifest structure: {type(manifest).__name__}"
    )


def infer_case_id(name: str):
    match = re.search(r"(case\d+)", str(name))
    return match.group(1) if match else None


def apply_spacing(
    target: Dict[str, Any],
    audit_case: Dict[str, Any],
):
    spacing_xyz = [
        float(v) for v in audit_case["spacing_xyz"]
    ]

    if len(spacing_xyz) != 3:
        raise ValueError(
            f"Invalid spacing_xyz for {audit_case['case_id']}: "
            f"{spacing_xyz}"
        )

    if not all(v > 0 for v in spacing_xyz):
        raise ValueError(
            f"Non-positive spacing for {audit_case['case_id']}: "
            f"{spacing_xyz}"
        )

    sx, sy, sz = spacing_xyz
    spacing_zyx = [sz, sy, sx]

    # Explicit fields.
    target["spacing_xyz"] = spacing_xyz
    target["spacing_zyx"] = spacing_zyx

    # Legacy compatibility. This field is stored as XYZ.
    target["spacing"] = spacing_xyz

    target["spacing_axis_order"] = {
        "spacing": "xyz",
        "spacing_xyz": "xyz",
        "spacing_zyx": "zyx",
    }

    target["spacing_source"] = audit_case.get(
        "spacing_source",
        "corresponding_btcv_nifti_header",
    )

    target["spacing_reference_case_id"] = audit_case.get(
        "mapped_btcv_case_id"
    )

    target["spacing_reference_path"] = audit_case.get(
        "btcv_reference_path"
    )

    if audit_case.get("orientation") is not None:
        target["orientation"] = audit_case["orientation"]


def main():
    for path in [AUDIT_PATH, MANIFEST_PATH, GEOMETRY_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    audit = load_json(AUDIT_PATH)
    manifest = load_json(MANIFEST_PATH)
    geometry = load_json(GEOMETRY_PATH)

    if audit.get("num_errors") != 0:
        raise RuntimeError(
            f"Audit contains errors: {audit.get('num_errors')}"
        )

    audit_cases = audit.get("cases")
    if not isinstance(audit_cases, dict):
        raise TypeError(
            "audit['cases'] must be a dictionary keyed by case ID"
        )

    if len(audit_cases) != 30:
        raise RuntimeError(
            f"Expected 30 audit cases, got {len(audit_cases)}"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for path in [MANIFEST_PATH, GEOMETRY_PATH]:
        backup = path + f".bak_before_spacing_fix_{timestamp}"
        shutil.copy2(path, backup)
        print(f"[BACKUP] {backup}")

    records = get_manifest_records(manifest)

    manifest_updated = 0
    missing_manifest_cases = []
    manifest_case_counts = Counter()
    geometry_key_to_case = {}

    for rec in records:
        case_id = str(rec.get("case_id", "")).strip()

        if not case_id:
            case_id = infer_case_id(rec.get("slice_name", ""))

        if not case_id:
            raise RuntimeError(
                f"Cannot determine case_id for manifest record: {rec}"
            )

        if case_id not in audit_cases:
            missing_manifest_cases.append(case_id)
            continue

        apply_spacing(rec, audit_cases[case_id])
        manifest_case_counts[case_id] += 1
        manifest_updated += 1

        geometry_key = rec.get(
            "geometry_key",
            rec.get("slice_name"),
        )
        if geometry_key:
            geometry_key_to_case[str(geometry_key)] = case_id

    if missing_manifest_cases:
        raise RuntimeError(
            "Manifest contains cases missing from audit: "
            f"{sorted(set(missing_manifest_cases))}"
        )

    geometry_updated = 0
    unresolved_geometry = []

    if not isinstance(geometry, dict):
        raise TypeError(
            f"geometry_meta.json must be dict, got "
            f"{type(geometry).__name__}"
        )

    for key, geom in geometry.items():
        if not isinstance(geom, dict):
            continue

        case_id = geom.get("case_id")

        if not case_id:
            case_id = geometry_key_to_case.get(str(key))

        if not case_id:
            case_id = infer_case_id(str(key))

        if not case_id:
            unresolved_geometry.append(str(key))
            continue

        case_id = str(case_id)

        if case_id not in audit_cases:
            unresolved_geometry.append(str(key))
            continue

        geom["case_id"] = case_id
        apply_spacing(geom, audit_cases[case_id])
        geometry_updated += 1

    if unresolved_geometry:
        raise RuntimeError(
            "Could not map geometry records to audited cases. "
            f"Examples: {unresolved_geometry[:10]}"
        )

    # Verify slice counts against audited volume depth.
    count_errors = []

    for case_id, audit_case in audit_cases.items():
        expected = int(audit_case["synapse_num_slices"])
        actual = int(manifest_case_counts.get(case_id, 0))

        if expected != actual:
            count_errors.append(
                {
                    "case_id": case_id,
                    "expected_slices": expected,
                    "manifest_slices": actual,
                }
            )

    if count_errors:
        raise RuntimeError(
            "Manifest slice counts do not match audited volumes:\n"
            + json.dumps(count_errors, indent=2)
        )

    atomic_save_json(manifest, MANIFEST_PATH)
    atomic_save_json(geometry, GEOMETRY_PATH)

    print()
    print("[PASS] Synapse spacing metadata repaired")
    print(f"manifest records updated: {manifest_updated}")
    print(f"geometry records updated: {geometry_updated}")
    print(f"cases validated: {len(audit_cases)}")
    print("axis convention:")
    print("  spacing_xyz = [x, y, z]")
    print("  spacing_zyx = [z, y, x]")
    print()
    print("case0001 expected:")
    print("  spacing_xyz = [0.66796875, 0.66796875, 3.0]")
    print("  spacing_zyx = [3.0, 0.66796875, 0.66796875]")


if __name__ == "__main__":
    main()
