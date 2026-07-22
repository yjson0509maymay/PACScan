"""Create a small synthetic 3D T2-like NIfTI for UI demonstration only."""

from pathlib import Path

import nibabel as nib
import numpy as np


root = Path(__file__).resolve().parents[1]
shape = (96, 112, 88)
x, y, z = np.meshgrid(
    np.linspace(-1, 1, shape[0]),
    np.linspace(-1, 1, shape[1]),
    np.linspace(-1, 1, shape[2]),
    indexing="ij",
)
brain = (x / .78) ** 2 + (y / .9) ** 2 + (z / .72) ** 2
volume = np.zeros(shape, dtype=np.float32)
volume[brain < 1] = 70 + 22 * (1 - brain[brain < 1])
white = (x / .58) ** 2 + (y / .7) ** 2 + (z / .55) ** 2
volume[white < 1] += 24
ventricles = (((x - .13) / .11) ** 2 + (y / .16) ** 2 + (z / .22) ** 2 < 1) | (((x + .13) / .11) ** 2 + (y / .16) ** 2 + (z / .22) ** 2 < 1)
volume[ventricles] = 18
rng = np.random.default_rng(42)
volume[brain < 1] += rng.normal(0, 2.2, np.count_nonzero(brain < 1))
affine = np.diag([1.2, 1.2, 1.5, 1.0])
nib.save(nib.Nifti1Image(volume, affine), root / "assets" / "sample_t2_mri.nii.gz")
print(root / "assets" / "sample_t2_mri.nii.gz")
