from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


class SyntheticPipelineTests(unittest.TestCase):
    def make_fold(self, root: Path, n_samples: int = 8) -> Path:
        fold = root / "processed" / "toy2d" / "fold_0"
        for d in [
            "native_npy/gts", "teacher_npy/imgs", "teacher_npy/gts",
            "student_npy/imgs", "student_npy/gts", "prompts", "meta",
            "pseudo_student/tri_train", "pseudo_teacher/tri_train",
        ]:
            (fold / d).mkdir(parents=True, exist_ok=True)
        manifest = []
        prompts = {}
        geometry = {}
        for idx in range(n_samples):
            name = f"sample_{idx:02d}.npy"
            gt = np.zeros((16, 16), dtype=np.uint8)
            gt[3:10, 4:12] = 1
            image = np.zeros((32, 32, 3), dtype=np.uint8)
            student_image = np.zeros((16, 16, 3), dtype=np.uint8)
            teacher_gt = np.zeros((32, 32), dtype=np.uint8)
            teacher_gt[6:20, 8:24] = 1
            np.save(fold / "native_npy/gts" / name, gt)
            np.save(fold / "teacher_npy/imgs" / name, image)
            np.save(fold / "teacher_npy/gts" / name, teacher_gt)
            np.save(fold / "student_npy/imgs" / name, student_image)
            np.save(fold / "student_npy/gts" / name, gt)
            baseline_tri = np.full((16, 16), 0, dtype=np.uint8)
            baseline_tri[3:10, 4:12] = 1
            np.save(fold / "pseudo_student/tri_train" / name, baseline_tri)
            np.save(fold / "pseudo_teacher/tri_train" / name, teacher_gt)
            manifest.append({
                "slice_name": name, "split": "train",
                "native_gt": f"native_npy/gts/{name}",
                "teacher_img": f"teacher_npy/imgs/{name}",
                "teacher_gt": f"teacher_npy/gts/{name}",
                "student_img": f"student_npy/imgs/{name}",
                "student_gt": f"student_npy/gts/{name}",
            })
            prompts[name] = {
                "split": "train", "fold": "fold_0", "empty_slice": False,
                "instances": [{
                    "label_id": 1, "component_id": 1,
                    "bbox_native": [4, 3, 11, 9],
                    "bbox_teacher": [8, 6, 22, 18],
                    "bbox": [8, 6, 22, 18],
                }],
            }
            geometry[name] = {
                "native_to_teacher": {
                    "orig_h": 16, "orig_w": 16, "target_h": 32, "target_w": 32,
                    "scale_x": 2.0, "scale_y": 2.0, "offset_x": 0, "offset_y": 0,
                    "resized_h": 32, "resized_w": 32,
                },
                "native_to_student": {
                    "orig_h": 16, "orig_w": 16, "target_h": 16, "target_w": 16,
                    "scale_x": 1.0, "scale_y": 1.0, "offset_x": 0, "offset_y": 0,
                    "resized_h": 16, "resized_w": 16,
                },
            }
        save_json(manifest, fold / "meta/manifest.json")
        save_json(geometry, fold / "meta/geometry_meta.json")
        save_json({"is_3d": False}, fold / "meta/split_meta.json")
        save_json({}, fold / "meta/label_meta.json")
        save_json(prompts, fold / "prompts/prompts_train.json")
        return fold

    def run_script(self, script: str, *args: str):
        subprocess.run([sys.executable, str(ROOT / script), *args], cwd=ROOT, check=True)

    def test_init_round0_pairs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            frozen_fold = self.make_fold(root / "frozen")
            idea_root = root / "idea" / "processed"
            self.run_script(
                "init_workspace.py",
                "--frozen_processed_root", str(frozen_fold.parents[1]),
                "--idea_processed_root", str(idea_root),
                "--dataset", "toy2d", "--shared_mode", "copy", "--boxonly_mode", "copy",
            )
            idea_fold = idea_root / "toy2d/fold_0"
            self.run_script(
                "select_round0.py", "--fold_root", str(idea_fold),
                "--dataset", "toy2d",
            )
            self.run_script(
                "build_finetune_pairs.py", "--fold_root", str(idea_fold),
                "--dataset", "toy2d", "--round", "0",
            )
            selection = json.loads((idea_fold / "rounds/idea1_hard_full_medsam_ft/round_00/selection/selection.json").read_text())
            pairs = json.loads((idea_fold / "rounds/idea1_hard_full_medsam_ft/round_00/finetune_pairs/pairs.json").read_text())
            self.assertEqual(len(selection["cumulative_full_slice_names"]), 1)
            self.assertEqual(len(pairs), 1)
            for row in pairs:
                self.assertTrue(Path(row["target_mask"]).is_file())

    def test_hard_selection_and_hybrid_assembly(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            frozen_fold = self.make_fold(root / "frozen", n_samples=40)
            idea_root = root / "idea" / "processed"
            self.run_script(
                "init_workspace.py",
                "--frozen_processed_root", str(frozen_fold.parents[1]),
                "--idea_processed_root", str(idea_root),
                "--dataset", "toy2d", "--shared_mode", "copy", "--boxonly_mode", "copy",
            )
            idea_fold = idea_root / "toy2d/fold_0"
            self.run_script(
                "select_round0.py", "--fold_root", str(idea_fold),
                "--dataset", "toy2d",
            )
            round0 = idea_fold / "rounds/idea1_hard_full_medsam_ft/round_00"
            current = json.loads((round0 / "selection/selection.json").read_text())
            box_names = current["remaining_box_slice_names"]
            # Current V2 selector uses image_min_iou as the
            # image-level difficulty metric.  Give every remaining Box
            # sample a diagnosis row: two hard samples and all others easy.
            scores = [0.1, 0.4] + [0.8] * (len(box_names) - 2)
            rows = [
                {"dataset": "toy2d", "round": 0, "slice_name": name, "case_id": "",
                 "num_instances": 1, "image_min_iou": score,
                 "empty_instance_count": int(score == 0.1),
                 "hard_candidate": int(score < 0.5)}
                for name, score in zip(box_names, scores)
            ]
            diag = round0 / "diagnosis"
            diag.mkdir(parents=True, exist_ok=True)
            import csv
            with (diag / "per_image_metrics.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader(); w.writerows(rows)
            self.run_script(
                "select_hard_samples.py", "--fold_root", str(idea_fold),
                "--dataset", "toy2d", "--round", "0",
            )
            round1_sel_path = idea_fold / "rounds/idea1_hard_full_medsam_ft/round_01/selection/selection.json"
            round1 = json.loads(round1_sel_path.read_text())
            self.assertEqual(len(round1["cumulative_full_slice_names"]), 2)
            remaining = round1["remaining_box_slice_names"]
            self.assertEqual(len(remaining), 38)
            box_final = idea_fold / "pseudo_student/tri_box_final_idea1_hard_full_medsam_ft"
            teacher_final = idea_fold / "pseudo_teacher/tri_box_final_idea1_hard_full_medsam_ft"
            box_final.mkdir(parents=True); teacher_final.mkdir(parents=True)
            for name in remaining:
                tri = np.zeros((16,16), dtype=np.uint8)
                tri[3:10,4:12] = 255
                tri[5:8,6:10] = 1
                np.save(box_final/name, tri)
                tri_t = np.zeros((32,32), dtype=np.uint8)
                tri_t[6:20,8:24] = 255
                tri_t[10:16,12:20] = 1
                np.save(teacher_final/name, tri_t)
            self.run_script(
                "assemble_supervision.py", "--fold_root", str(idea_fold),
                "--dataset", "toy2d", "--final_round", "1",
            )
            view_root = root / "views"
            self.run_script(
                "build_student_view.py", "--fold_root", str(idea_fold),
                "--view_root", str(view_root), "--dataset", "toy2d", "--build_boxonly",
            )
            self.run_script(
                "validate_data.py", "--fold_root", str(idea_fold),
                "--dataset", "toy2d", "--final_round", "1",
            )
            report = json.loads((idea_fold / "meta/method_validation_idea1_hard_full_medsam_ft.json").read_text())
            self.assertTrue(report["passed"])
            self.assertEqual(report["counts"]["full"], 2)
            self.assertEqual(report["counts"]["box"], 38)


if __name__ == "__main__":
    unittest.main()
