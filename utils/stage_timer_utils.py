import os
import json
import time
import math
import socket
from contextlib import ContextDecorator


def sec_to_hms(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60.0
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def write_stage_time(
    save_path: str,
    stage_name: str,
    dataset: str,
    fold: str,
    mode: str = "",
    split: str = "",
    status: str = "success",
    start_ts: float = None,
    end_ts: float = None,
    num_outputs: int = 0,
    notes: str = "",
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    duration_sec = max(0.0, float(end_ts - start_ts))
    obj = {
        "stage_name": stage_name,
        "dataset": dataset,
        "fold": fold,
        "mode": mode,
        "split": split,
        "status": status,
        "start_ts": float(start_ts),
        "end_ts": float(end_ts),
        "duration_sec": duration_sec,
        "duration_hms": sec_to_hms(duration_sec),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "num_outputs": int(num_outputs),
        "notes": notes,
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


class StageTimer(ContextDecorator):
    def __init__(
        self,
        save_path: str,
        stage_name: str,
        dataset: str,
        fold: str,
        mode: str = "",
        split: str = "",
    ):
        self.save_path = save_path
        self.stage_name = stage_name
        self.dataset = dataset
        self.fold = fold
        self.mode = mode
        self.split = split
        self.start_ts = None
        self.end_ts = None
        self.num_outputs = 0
        self.notes = ""
        self.status = "success"

    def __enter__(self):
        self.start_ts = time.time()
        return self

    def set_outputs(self, n: int):
        self.num_outputs = int(n)

    def set_notes(self, notes: str):
        self.notes = str(notes)

    def __exit__(self, exc_type, exc, tb):
        self.end_ts = time.time()
        if exc is not None:
            self.status = "failed"
            if not self.notes:
                self.notes = repr(exc)

        write_stage_time(
            save_path=self.save_path,
            stage_name=self.stage_name,
            dataset=self.dataset,
            fold=self.fold,
            mode=self.mode,
            split=self.split,
            status=self.status,
            start_ts=self.start_ts,
            end_ts=self.end_ts,
            num_outputs=self.num_outputs,
            notes=self.notes,
        )
        return False