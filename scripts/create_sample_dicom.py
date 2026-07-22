"""Create a synthetic T2-like DICOM series ZIP for PACScan UI testing."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid


root = Path(__file__).resolve().parents[1]
series_dir = root / "assets" / "sample_dicom_patient" / "T2_AXIAL"
series_dir.mkdir(parents=True, exist_ok=True)
study_uid, series_uid = generate_uid(), generate_uid()
rows = cols = 96
x, y = np.meshgrid(np.linspace(-1, 1, cols), np.linspace(-1, 1, rows))

for index in range(32):
    z = (index - 15.5) / 17
    brain = (x / .78) ** 2 + (y / .9) ** 2 + (z / .78) ** 2
    pixels = np.zeros((rows, cols), dtype=np.uint16)
    mask = brain < 1
    pixels[mask] = (850 + 500 * (1 - brain[mask])).astype(np.uint16)
    ventricles = (((x - .14) / .10) ** 2 + (y / .17) ** 2 < 1) | (((x + .14) / .10) ** 2 + (y / .17) ** 2 < 1)
    pixels[ventricles & mask] = 220

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationClassUID = generate_uid()
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.Modality = "MR"
    ds.PatientID = "PACSCAN-DEMO-001"
    ds.PatientName = "PACScan^Demo"
    ds.SeriesDescription = "T2 AXIAL"
    ds.ProtocolName = "T2_TSE_AX"
    ds.InstanceNumber = index + 1
    ds.Rows, ds.Columns = rows, cols
    ds.PixelSpacing = [1.2, 1.2]
    ds.SliceThickness = 2.0
    ds.SpacingBetweenSlices = 2.0
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.ImagePositionPatient = [0, 0, index * 2.0]
    ds.SliceLocation = index * 2.0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.PixelRepresentation = 0
    ds.RescaleIntercept = 0
    ds.RescaleSlope = 1
    ds.RepetitionTime = 4000
    ds.EchoTime = 90
    ds.PixelData = pixels.tobytes()
    ds.save_as(series_dir / f"T2_{index + 1:04d}.dcm", enforce_file_format=True)

zip_path = root / "assets" / "PACScan_sample_DICOM_folder.zip"
with ZipFile(zip_path, "w", ZIP_DEFLATED) as archive:
    for path in sorted(series_dir.glob("*.dcm")):
        archive.write(path, f"PACScan_sample_patient/T2_AXIAL/{path.name}")
print(zip_path)
