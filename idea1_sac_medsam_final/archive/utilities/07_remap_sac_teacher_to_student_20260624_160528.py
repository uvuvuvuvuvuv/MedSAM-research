import argparse
import json
import os
import glob

import cv2
import numpy as np


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_mask(path):
    m = np.load(path)
    if m.ndim == 3:
        if m.shape[-1] == 1:
            m = m[..., 0]
        elif m.shape[0] == 1:
            m = m[0]
    return m.astype(np.uint8)


def restore_teacher_to_native(mask_teacher, geom):
    if "teacher_to_native" in geom:
        g = geom["teacher_to_native"]
        crop_h = int(g.get("crop_h", g.get("new_h", mask_teacher.shape[0])))
        crop_w = int(g.get("crop_w", g.get("new_w", mask_teacher.shape[1])))
        orig_h = int(g.get("to_h", geom.get("orig_h", geom.get("native_hw", [0, 0])[0])))
        orig_w = int(g.get("to_w", geom.get("orig_w", geom.get("native_hw", [0, 0])[1])))
    else:
        crop_h = int(geom["new_h"])
        crop_w = int(geom["new_w"])
        orig_h = int(geom["orig_h"])
        orig_w = int(geom["orig_w"])

    cropped = mask_teacher[:crop_h, :crop_w]
    native = cv2.resize(cropped, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return native.astype(np.uint8)


def restore_native_mask_to_student(mask_native, geom):
    if "native_to_student" not in geom:
        raise KeyError("geometry meta missing native_to_student")

    g = geom["native_to_student"]
    target_h = int(g["target_h"])
    target_w = int(g["target_w"])
    new_h = int(g["new_h"])
    new_w = int(g["new_w"])
    offset_x = int(g.get("offset_x", 0))
    offset_y = int(g.get("offset_y", 0))

    resized = cv2.resize(
        mask_native.astype(np.uint8),
        (new_w, new_h),
        interpolation=cv2.INTER_NEAREST,
    )

    canvas = np.zeros((target_h, target_w), dtype=np.uint8)
    canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold_root", required=True)
    parser.add_argument("--method", default="idea1_sac_medsam_final")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    fold = args.fold_root
    meta_dir = os.path.join(fold, "meta")

    manifest = load_json(os.path.join(meta_dir, "manifest.json"))
    geom = load_json(os.path.join(meta_dir, "geometry_meta.json"))
    split_records = load_json(os.path.join(meta_dir, f"full_box_split_{args.method}.json"))

    manifest_by_slice = {x["slice_name"]: x for x in manifest}
    split_names = [x["slice_name"] for x in split_records]

    teacher_dir = os.path.join(fold, "pseudo_teacher", f"tri_train_{args.method}")
    student_dir = os.path.join(fold, "pseudo_student", f"tri_train_{args.method}")
    os.makedirs(student_dir, exist_ok=True)

    if args.overwrite:
        for f in glob.glob(os.path.join(student_dir, "*.npy")):
            os.remove(f)

    n = 0
    for name in split_names:
        item = manifest_by_slice[name]
        if item.get("split") != "train":
            raise RuntimeError(f"non-train leakage: {name}")

        teacher_path = os.path.join(teacher_dir, name)
        if not os.path.exists(teacher_path):
            raise FileNotFoundError(teacher_path)

        g = geom.get(name, geom.get(item.get("geometry_key", name), {}))

        tri_teacher = load_mask(teacher_path)
        tri_native = restore_teacher_to_native(tri_teacher, g)
        tri_student = restore_native_mask_to_student(tri_native, g)

        out_path = os.path.join(student_dir, name)
        np.save(out_path, tri_student.astype(np.uint8))
        n += 1

        if n % 50 == 0:
            print(f"remapped {n}/{len(split_names)}")

    print("[OK] remap teacher pseudo -> student pseudo done")
    print("teacher_dir =", teacher_dir)
    print("student_dir =", student_dir)
    print("num =", n)


if __name__ == "__main__":
    main()
