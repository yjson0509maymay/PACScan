"""Run the bundled synthetic DICOM through the real local pipeline."""

from pathlib import Path
from zipfile import ZipFile
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from local_pipeline import run_local_pipeline
from preprocessing_adapter import inspect_dicom_folder


sample = ROOT / "assets" / "PACScan_sample_DICOM_folder.zip"
with ZipFile(sample) as archive:
    files = [(name, archive.read(name)) for name in archive.namelist() if not name.endswith("/")]
scan = inspect_dicom_folder(files)
print(f"scan_valid={scan.valid} dicom={scan.dicom_files} series={scan.series_count}", flush=True)
result = run_local_pipeline(
    files,
    scan,
    ROOT,
    progress=lambda value, label: print(f"{value:3d}% {label}", flush=True),
)
print(f"run_id={result['run_id']}", flush=True)
print(f"mode={result['pipeline_mode']} shape={result['final_shape']} elapsed={result['elapsed_sec']}", flush=True)
