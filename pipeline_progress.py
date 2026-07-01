#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline_progress.py

Small JSON/CSV/HTML progress writer for the MedSAM -> Swin-UMamba baseline pipeline.
It has no third-party dependencies and is safe to import from:
  - utils/processed.py
  - generate_prompts.py
  - generate_pseudo_labels.py

Output layout:
  <progress_root>/
    index.html
    index.json
    summary.csv
    events.csv
    <stage>/<dataset>/<fold>/<split>.json
"""

from __future__ import annotations

import csv
import html
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except Exception:
        return None


def _safe_name(x: Any) -> str:
    s = str(x if x is not None else "unknown").strip()
    if not s:
        s = "unknown"
    return "".join(ch if (ch.isalnum() or ch in {"_", "-", "."}) else "_" for ch in s)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_json(obj: Any, path: Path) -> None:
    _ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fmt_seconds(seconds: Optional[float]) -> str:
    if seconds is None:
        return ""
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _flatten_stats(stats: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not isinstance(stats, dict):
        return out
    for k, v in stats.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[str(k)] = v
    return out


class ProgressWriter:
    def __init__(
        self,
        progress_root: Optional[str | Path],
        stage: str,
        dataset: str,
        fold: str = "fold_0",
        split: str = "all",
        enabled: bool = True,
    ):
        self.enabled = bool(enabled and progress_root)
        self.progress_root = Path(progress_root) if progress_root else Path(".")
        self.stage = _safe_name(stage)
        self.dataset = _safe_name(dataset)
        self.fold = _safe_name(fold)
        self.split = _safe_name(split)
        self.record_path = self.progress_root / self.stage / self.dataset / self.fold / f"{self.split}.json"
        self.events_path = self.progress_root / "events.csv"
        self._last_write_monotonic = 0.0

    def start(self, total: Optional[int] = None, message: str = "", stats: Optional[Dict[str, Any]] = None) -> None:
        self.update(status="running", current=0, total=total, message=message, stats=stats, force=True)

    def update(
        self,
        status: str = "running",
        current: Optional[int] = None,
        total: Optional[int] = None,
        message: str = "",
        stats: Optional[Dict[str, Any]] = None,
        force: bool = False,
        min_interval_seconds: float = 2.0,
    ) -> None:
        if not self.enabled:
            return
        now_mono = time.monotonic()
        if not force and (now_mono - self._last_write_monotonic) < float(min_interval_seconds):
            return
        self._last_write_monotonic = now_mono

        now_iso = _utc_now_iso()
        old: Dict[str, Any] = {}
        if self.record_path.exists():
            try:
                old = _load_json(self.record_path)
            except Exception:
                old = {}

        started_at = old.get("started_at") or now_iso
        finished_at = now_iso if status in {"success", "failed", "skipped"} else old.get("finished_at")
        started_dt = _parse_iso(started_at)
        now_dt = _parse_iso(now_iso)
        elapsed_seconds = None
        if started_dt is not None and now_dt is not None:
            elapsed_seconds = max(0.0, (now_dt - started_dt).total_seconds())

        prev_stats = old.get("stats") if isinstance(old.get("stats"), dict) else {}
        merged_stats = dict(prev_stats)
        if isinstance(stats, dict):
            merged_stats.update(stats)

        if current is None:
            current = old.get("current", 0)
        if total is None:
            total = old.get("total", 0)
        try:
            current_i = int(current or 0)
        except Exception:
            current_i = 0
        try:
            total_i = int(total or 0)
        except Exception:
            total_i = 0
        percent = None
        if total_i > 0:
            percent = round(min(100.0, max(0.0, current_i * 100.0 / total_i)), 2)

        record: Dict[str, Any] = {
            "stage": self.stage,
            "dataset": self.dataset,
            "fold": self.fold,
            "split": self.split,
            "status": str(status),
            "current": current_i,
            "total": total_i,
            "percent": percent,
            "message": str(message or old.get("message", "")),
            "started_at": started_at,
            "last_update_at": now_iso,
            "finished_at": finished_at,
            "elapsed_seconds": elapsed_seconds,
            "elapsed_hms": _fmt_seconds(elapsed_seconds),
            "stats": merged_stats,
        }
        _save_json(record, self.record_path)
        self._append_event(record)
        self.rebuild_index(self.progress_root)

    def finish(self, status: str = "success", message: str = "", stats: Optional[Dict[str, Any]] = None, current: Optional[int] = None, total: Optional[int] = None) -> None:
        if current is None and self.record_path.exists():
            try:
                old = _load_json(self.record_path)
                current = old.get("current", 0)
                total = old.get("total", total)
            except Exception:
                pass
        self.update(status=status, current=current, total=total, message=message, stats=stats, force=True)

    def _append_event(self, record: Dict[str, Any]) -> None:
        _ensure_dir(self.events_path.parent)
        flat = _flatten_stats(record.get("stats"))
        fieldnames = [
            "time", "stage", "dataset", "fold", "split", "status",
            "current", "total", "percent", "elapsed_seconds", "elapsed_hms", "message",
        ]
        extra_keys = [f"stats.{k}" for k in sorted(flat.keys())]
        all_fields = fieldnames + extra_keys
        exists = self.events_path.exists()
        row = {
            "time": record.get("last_update_at", ""),
            "stage": record.get("stage", ""),
            "dataset": record.get("dataset", ""),
            "fold": record.get("fold", ""),
            "split": record.get("split", ""),
            "status": record.get("status", ""),
            "current": record.get("current", ""),
            "total": record.get("total", ""),
            "percent": record.get("percent", ""),
            "elapsed_seconds": record.get("elapsed_seconds", ""),
            "elapsed_hms": record.get("elapsed_hms", ""),
            "message": record.get("message", ""),
        }
        for k, v in flat.items():
            row[f"stats.{k}"] = v
        # If keys grow over time, summary.csv/index.json remain authoritative; events.csv is an append log.
        write_header = (not exists) or self.events_path.stat().st_size == 0
        with open(self.events_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    @staticmethod
    def collect_records(progress_root: str | Path) -> list[Dict[str, Any]]:
        root = Path(progress_root)
        if not root.exists():
            return []
        records: list[Dict[str, Any]] = []
        for p in sorted(root.glob("*/*/*/*.json")):
            if p.name in {"index.json"}:
                continue
            try:
                obj = _load_json(p)
                if isinstance(obj, dict) and "stage" in obj and "dataset" in obj:
                    records.append(obj)
            except Exception:
                continue
        records.sort(key=lambda r: (str(r.get("dataset", "")), str(r.get("stage", "")), str(r.get("fold", "")), str(r.get("split", ""))))
        return records

    @staticmethod
    def rebuild_index(progress_root: str | Path) -> None:
        root = Path(progress_root)
        _ensure_dir(root)
        records = ProgressWriter.collect_records(root)
        _save_json({"updated_at": _utc_now_iso(), "records": records}, root / "index.json")
        ProgressWriter._write_summary_csv(root, records)
        ProgressWriter._write_html(root, records)

    @staticmethod
    def _write_summary_csv(root: Path, records: list[Dict[str, Any]]) -> None:
        fields = [
            "stage", "dataset", "fold", "split", "status", "current", "total", "percent",
            "elapsed_seconds", "elapsed_hms", "started_at", "last_update_at", "finished_at", "message",
        ]
        extra = sorted({f"stats.{k}" for r in records for k in _flatten_stats(r.get("stats")).keys()})
        with open(root / "summary.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields + extra, extrasaction="ignore")
            writer.writeheader()
            for r in records:
                row = {k: r.get(k, "") for k in fields}
                for k, v in _flatten_stats(r.get("stats")).items():
                    row[f"stats.{k}"] = v
                writer.writerow(row)

    @staticmethod
    def _write_html(root: Path, records: list[Dict[str, Any]]) -> None:
        status_class = {
            "running": "running",
            "success": "success",
            "failed": "failed",
            "skipped": "skipped",
        }
        rows = []
        for r in records:
            status = str(r.get("status", ""))
            cls = status_class.get(status, "")
            percent = r.get("percent")
            pct_str = "" if percent is None else f"{float(percent):.1f}%"
            total = int(r.get("total") or 0)
            current = int(r.get("current") or 0)
            bar_width = 0 if total <= 0 else min(100, max(0, current * 100.0 / total))
            stats = _flatten_stats(r.get("stats"))
            stat_text = ", ".join(f"{html.escape(str(k))}={html.escape(str(v))}" for k, v in stats.items())
            rows.append(f"""
<tr class='{cls}'>
<td>{html.escape(str(r.get('dataset','')))}</td>
<td>{html.escape(str(r.get('stage','')))}</td>
<td>{html.escape(str(r.get('fold','')))}</td>
<td>{html.escape(str(r.get('split','')))}</td>
<td><span class='badge {cls}'>{html.escape(status)}</span></td>
<td><div class='bar'><div style='width:{bar_width:.1f}%'></div></div><span>{pct_str}</span></td>
<td>{current}/{total}</td>
<td>{html.escape(str(r.get('elapsed_hms','')))}</td>
<td>{html.escape(str(r.get('last_update_at','')))}</td>
<td>{html.escape(str(r.get('message','')))}</td>
<td class='stats'>{stat_text}</td>
</tr>""")
        body = "\n".join(rows)
        updated = _utc_now_iso()
        page = f"""<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'>
<meta http-equiv='refresh' content='10'>
<title>Baseline pipeline progress</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 20px; background: #f7f7f8; color: #222; }}
h1 {{ margin-bottom: 4px; }}
.small {{ color: #666; margin-bottom: 16px; }}
table {{ border-collapse: collapse; width: 100%; background: white; box-shadow: 0 1px 4px #ddd; }}
th, td {{ border-bottom: 1px solid #eee; padding: 8px; text-align: left; vertical-align: top; font-size: 13px; }}
th {{ background: #fafafa; position: sticky; top: 0; }}
.badge {{ padding: 3px 7px; border-radius: 10px; color: white; font-size: 12px; }}
.badge.running {{ background: #2b6cb0; }}
.badge.success {{ background: #2f855a; }}
.badge.failed {{ background: #c53030; }}
.badge.skipped {{ background: #718096; }}
tr.failed {{ background: #fff5f5; }}
tr.running {{ background: #ebf8ff; }}
.bar {{ width: 120px; height: 10px; background: #e2e8f0; border-radius: 6px; overflow: hidden; display: inline-block; margin-right: 6px; }}
.bar div {{ height: 100%; background: #3182ce; }}
.stats {{ max-width: 420px; word-break: break-word; color: #444; }}
</style>
</head>
<body>
<h1>Baseline pipeline progress</h1>
<div class='small'>Updated at {html.escape(updated)}. This page auto-refreshes every 10 seconds.</div>
<table>
<thead><tr><th>Dataset</th><th>Stage</th><th>Fold</th><th>Split</th><th>Status</th><th>Progress</th><th>Count</th><th>Elapsed</th><th>Last update</th><th>Message</th><th>Stats</th></tr></thead>
<tbody>
{body}
</tbody>
</table>
</body>
</html>
"""
        with open(root / "index.html", "w", encoding="utf-8") as f:
            f.write(page)
