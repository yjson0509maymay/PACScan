"""Local bridge from PACScan to BRAINTENSOR ref21order_v1 preprocessing."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from preprocessing_adapter import preprocess_nifti


APP_VERSION = "1.2.0"
PIPELINE_VERSION = "ref21order_v1"
DEFAULT_SCRIPT = Path(r"E:\해커톤\BRAINTENSOR\01_Preprocessing\스크립트\preparing_ref21order_v1.py")
DEFAULT_ATLAS = Path(r"E:\ppmi_dti\derivatives\templates\templates\pd25_20170213\PD25-T1MPRAGE-template-1mm.nii.gz")
DEFAULT_WSL_DISTRO = "Ubuntu-24.04"


@dataclass(frozen=True)
class LocalPipelineStatus:
    ready: bool
    script: str
    atlas: str
    fsl_bet: bool
    antspyx: bool
    message: str


def _configured_path(variable: str, default: Path) -> Path:
    return Path(os.environ.get(variable, str(default))).expanduser()


def _has_antspyx() -> bool:
    return importlib.util.find_spec("ants") is not None


def _has_fsl_bet() -> bool:
    if shutil.which("bet"):
        return True
    if not shutil.which("wsl"):
        return False
    try:
        distro = os.environ.get("PACSCAN_WSL_DISTRO", DEFAULT_WSL_DISTRO)
        check = subprocess.run(
            ["wsl", "-d", distro, "bash", "-lc", "command -v bet >/dev/null 2>&1"],
            timeout=8,
            check=False,
        )
        return check.returncode == 0
    except Exception:
        return False


def local_pipeline_status() -> LocalPipelineStatus:
    script = _configured_path("PACSCAN_BRAINTENSOR_SCRIPT", DEFAULT_SCRIPT)
    atlas = _configured_path("PACSCAN_PD25_ATLAS", DEFAULT_ATLAS)
    fsl_bet = _has_fsl_bet()
    antspyx = _has_antspyx()
    missing = []
    if not script.is_file():
        missing.append("BRAINTENSOR 전처리 스크립트")
    if not atlas.is_file():
        missing.append("PD25 아틀라스")
    if not fsl_bet:
        missing.append("FSL BET")
    if not antspyx:
        missing.append("antspyx")
    ready = not missing
    message = "실제 로컬 전처리 준비 완료" if ready else "필요 환경: " + ", ".join(missing)
    return LocalPipelineStatus(ready, str(script), str(atlas), fsl_bet, antspyx, message)


def _git_revision(repo: Path) -> str:
    try:
        head = (repo / ".git" / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref: "):
            ref = repo / ".git" / head[5:]
            return ref.read_text(encoding="utf-8").strip()[:12]
        return head[:12]
    except Exception:
        return "unknown"


def _utc_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _append_index(path: Path, row: dict) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _load_pipeline(script: Path):
    spec = importlib.util.spec_from_file_location("braintensor_ref21order_v1", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"전처리 코드를 불러올 수 없습니다: {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _configure_wsl_bridge(module) -> None:
    """Force the installed FSL distro instead of relying on WSL's default distro."""
    distro = os.environ.get("PACSCAN_WSL_DISTRO", DEFAULT_WSL_DISTRO)

    def run_wsl_bash(command: str, step_name: str):
        result = subprocess.run(
            ["wsl", "-d", distro, "bash", "-lc", command],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False,
        )
        if result.returncode != 0:
            message = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"{step_name} failed(WSL {distro}): {message}")
        return result

    module._run_wsl_bash = run_wsl_bash


def run_local_pipeline(
    files: list[tuple[str, bytes]],
    folder_scan,
    app_root: Path,
    progress: Callable[[int, str], None] | None = None,
) -> dict:
    """Run GitHub BRAINTENSOR v1 and persist a complete per-run audit trail."""
    status = local_pipeline_status()
    if not status.ready:
        raise RuntimeError(status.message)

    started = time.perf_counter()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{stamp}_{hashlib.sha256(os.urandom(16)).hexdigest()[:8]}"
    runs_root = app_root / "results" / "runs"
    run_dir = runs_root / run_id
    dicom_dir = run_dir / "input_dicom"
    output_dir = run_dir / "artifacts"
    dicom_dir.mkdir(parents=True, exist_ok=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    steps_path = run_dir / "steps.jsonl"
    script = Path(status.script)
    atlas = Path(status.atlas)
    patient_hash = hashlib.sha256(f"pacscan:{folder_scan.patient_id}".encode()).hexdigest()[:16]
    braintensor_repo = script.parents[2]
    run_record = {
        "schema_version": 1,
        "run_id": run_id,
        "started_at": _utc_now(),
        "finished_at": None,
        "app_version": APP_VERSION,
        "app_commit": _git_revision(app_root),
        "pipeline": PIPELINE_VERSION,
        "pipeline_script": str(script),
        "pipeline_commit": _git_revision(braintensor_repo),
        "patient_hash": patient_hash,
        "input_type": "dicom_directory",
        "dicom_count": folder_scan.selected_files,
        "series_count": folder_scan.series_count,
        "selected_series": folder_scan.selected_description,
        "atlas": str(atlas),
        "bet_frac": 0.5,
        "normalization": "minmax",
        "bias_correction": False,
        "target_shape": [56, 56, 56],
        "model_mode": "demo_not_connected",
        "status": "running",
        "elapsed_sec": None,
    }
    _write_json(run_dir / "run.json", run_record)

    selected = []
    for index, (name, payload) in enumerate(files):
        try:
            import io
            import pydicom
            ds = pydicom.dcmread(io.BytesIO(payload), stop_before_pixels=True, force=True)
            if str(getattr(ds, "SeriesInstanceUID", "")) != folder_scan.selected_uid:
                continue
        except Exception:
            continue
        target = dicom_dir / f"slice_{index:05d}.dcm"
        target.write_bytes(payload)
        selected.append(target)
    if len(selected) < 2:
        raise RuntimeError("선택된 T2 DICOM 슬라이스가 부족합니다.")

    module = _load_pipeline(script)
    _configure_wsl_bridge(module)
    stage_specs = [
        ("convert_dicom_to_nifti", "dicom_to_nifti", 15, "DICOM → NIfTI 변환"),
        ("run_skull_stripping_bet", "bet", 34, "FSL BET 뇌 추출"),
        ("run_registration_antspy", "registration", 58, "ANTsPy PD25 정합"),
        ("normalize_minmax", "normalization", 74, "Min-Max 정규화"),
        ("resize_nifti", "resize", 90, "56×56×56 리사이즈"),
    ]
    for function_name, step_name, percent, label in stage_specs:
        original = getattr(module, function_name)

        def wrapper(*args, _original=original, _step=step_name, _percent=percent, _label=label, **kwargs):
            step_started = time.perf_counter()
            event = {"run_id": run_id, "step": _step, "started_at": _utc_now(), "status": "running"}
            _append_jsonl(steps_path, event)
            if progress:
                progress(_percent, _label)
            try:
                value = _original(*args, **kwargs)
                event.update(status="success", finished_at=_utc_now(), elapsed_sec=round(time.perf_counter() - step_started, 3))
                _append_jsonl(steps_path, event)
                return value
            except Exception as exc:
                event.update(status="failed", finished_at=_utc_now(), elapsed_sec=round(time.perf_counter() - step_started, 3), error=str(exc))
                _append_jsonl(steps_path, event)
                raise

        setattr(module, function_name, wrapper)
        if function_name == "normalize_minmax":
            module.NORMALIZATION_FUNCS["minmax"] = wrapper

    name = f"pacscan_{run_id}"
    try:
        info = module.preprocess_one(
            dicom_dir=str(dicom_dir), output_dir=str(output_dir), name=name,
            atlas_path=str(atlas), normalization="minmax", enable_bias_correction=False,
            target_shape=(56, 56, 56), bet_frac=0.5, augment_count=0,
            seed=42, keep_intermediate=True,
        )
        final_path = output_dir / f"{name}.nii.gz"
        raw_path = output_dir / "_work" / name / f"{name}_00_raw.nii.gz"
        final_bytes = final_path.read_bytes()
        raw_bytes = raw_path.read_bytes()
        original_display = preprocess_nifti(raw_bytes, raw_path.name)
        final_display = preprocess_nifti(final_bytes, final_path.name)
        prep = final_display
        prep.update({
            "original_views": original_display["original_views"],
            "original_shape": original_display["original_shape"],
            "processed_views": final_display["original_views"],
            "output_bytes": final_bytes,
            "output_name": final_path.name,
            "pipeline_mode": "local_full",
            "pipeline_version": PIPELINE_VERSION,
            "run_id": run_id,
            "elapsed_sec": round(time.perf_counter() - started, 3),
        })
        checksum = hashlib.sha256(final_bytes).hexdigest()
        artifacts = {
            "run_id": run_id,
            "items": [{
                "artifact_type": "final_nifti", "filename": final_path.name,
                "relative_path": str(final_path.relative_to(app_root)),
                "shape": [56, 56, 56], "size_bytes": len(final_bytes), "sha256": checksum,
            }],
        }
        _write_json(run_dir / "artifacts.json", artifacts)
        run_record.update(status="success", finished_at=_utc_now(), elapsed_sec=prep["elapsed_sec"], result=info)
        _write_json(run_dir / "run.json", run_record)
        _append_index(runs_root / "run_index.csv", {
            "run_id": run_id, "started_at": run_record["started_at"], "app_version": APP_VERSION,
            "app_commit": run_record["app_commit"], "pipeline_version": PIPELINE_VERSION,
            "pipeline_commit": run_record["pipeline_commit"], "status": "success",
            "elapsed_sec": prep["elapsed_sec"], "dicom_count": len(selected),
            "final_shape": "56x56x56", "output_sha256": checksum,
        })
        if progress:
            progress(100, "실제 전처리 및 로그 저장 완료")
        return prep
    except Exception as exc:
        elapsed = round(time.perf_counter() - started, 3)
        run_record.update(status="failed", finished_at=_utc_now(), elapsed_sec=elapsed)
        _write_json(run_dir / "run.json", run_record)
        _write_json(run_dir / "error.json", {"run_id": run_id, "error_type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
        _append_index(runs_root / "run_index.csv", {
            "run_id": run_id, "started_at": run_record["started_at"], "app_version": APP_VERSION,
            "app_commit": run_record["app_commit"], "pipeline_version": PIPELINE_VERSION,
            "pipeline_commit": run_record["pipeline_commit"], "status": "failed",
            "elapsed_sec": elapsed, "dicom_count": len(selected), "final_shape": "", "output_sha256": "",
        })
        raise
