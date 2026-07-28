# -*- coding: utf-8 -*-
"""cca_ridge_J.py - 동료 J의 CCA(정준상관분석) 구현 이식.

원본: C:\\Users\\user\\Downloads\\Jfile\\cca_paper_performance_v3_J.py
(RCCASummary/RidgeCCA/ViewReducer/IndependentPCARCCA 클래스만 그대로 옮김 -
그리드서치/보고서 생성 등 나머지 실험 러너 코드는 이 프로젝트에서 안 씀).

이 프로젝트(BRAINTENSOR)의 자체 CCA 구현(cca_feature_fusion.py, 논문
Algorithm1/Eq6/Eq7 문자 그대로 재현 + 상한 100 고정)과 달리, J의 구현은:
- FC-3(CNN)과 FC-4(ResNet)에 서로 다른 PCA 차원을 독립적으로 적용(pca_x/pca_y)
- Ridge 정규화 CCA(ridge=0이면 논문과 동일한 순수 CCA)
- 반복 층화 교차검증으로 하이퍼파라미터를 개발셋(85%)에서만 선택하고
  최종 15% held-out 테스트는 선택에 전혀 관여하지 않음
등 방법론적으로 더 엄격함(이 프로젝트에서 "동료 J 결과가 더 신뢰할 만하다"고
여러 번 언급된 이유). "4개 모델(CNN+ResNet+CCA+WOA) 앙상블"을 실제 서비스에
연결하기로 한 결정에 따라, CCA 단계만 J의 이 구현으로 교체한다(ResNet은 J의
학습된 체크포인트가 없어 우리 체크포인트 유지, WOA도 우리 구현 유지).

동료 J가 그리드서치로 찾은 최종 채택 설정(cca_paper_performance_v3_results_J.xlsx
Best_Config 시트, 우리 자체 CNN/ResNet 특징에도 참고용으로 재사용):
pca_x=40, pca_y=10, cca_components=10, ridge=0.1, standardize=False, fusion='concat'
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class RCCASummary:
    ridge: float
    rank_x: int
    rank_y: int
    nonzero_components: int
    available_components: int
    eigenvalue_tolerance: float
    mean_train_corr: float
    min_train_corr: float
    max_train_corr: float
    corr_ge_0999: int


class RidgeCCA:
    """Compact-SVD CCA/RCCA suitable for p,q >> n.

    ridge=0:
        paper-like CCA on the non-zero covariance subspace.
    ridge>0:
        L2-regularized within-view covariance inverses.
    """

    def __init__(self, ridge: float = 0.0, eps: float = 1e-10, eigenvalue_tol: float = 1e-12):
        if ridge < 0:
            raise ValueError("ridge must be >=0")
        self.ridge = float(ridge)
        self.eps = float(eps)
        self.eigenvalue_tol = float(eigenvalue_tol)
        self.mean_x: Optional[np.ndarray] = None
        self.mean_y: Optional[np.ndarray] = None
        self.wx: Optional[np.ndarray] = None
        self.wy: Optional[np.ndarray] = None
        self.corr_: Optional[np.ndarray] = None
        self.eig_: Optional[np.ndarray] = None
        self.summary_: Optional[RCCASummary] = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgeCCA":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
            raise ValueError(f"CCA inputs must be 2D with matching N: {x.shape} vs {y.shape}")
        if x.shape[0] < 3:
            raise ValueError("CCA requires >=3 samples")

        self.mean_x = x.mean(axis=0)
        self.mean_y = y.mean(axis=0)
        xc = x - self.mean_x
        yc = y - self.mean_y
        n = x.shape[0]

        ux, sx, vtx = np.linalg.svd(xc, full_matrices=False)
        uy, sy, vty = np.linalg.svd(yc, full_matrices=False)

        tol_x = self.eps * max(xc.shape) * (float(sx[0]) if sx.size else 1.0)
        tol_y = self.eps * max(yc.shape) * (float(sy[0]) if sy.size else 1.0)
        keep_x = sx > tol_x
        keep_y = sy > tol_y
        ux, sx, vtx = ux[:, keep_x], sx[keep_x], vtx[keep_x]
        uy, sy, vty = uy[:, keep_y], sy[keep_y], vty[keep_y]

        rank_x, rank_y = len(sx), len(sy)
        if rank_x < 1 or rank_y < 1:
            raise ValueError("One CCA view has zero numerical rank")

        scale = math.sqrt(max(n - 1, 1))
        lam_x = (sx / scale) ** 2
        lam_y = (sy / scale) ** 2

        shrink_x = (sx / scale) / np.sqrt(lam_x + self.ridge)
        shrink_y = (sy / scale) / np.sqrt(lam_y + self.ridge)
        whitened_cross = shrink_x[:, None] * (ux.T @ uy) * shrink_y[None, :]

        p, corr, qt = np.linalg.svd(whitened_cross, full_matrices=False)
        corr = np.clip(corr, 0.0, 1.0)
        eig = corr ** 2

        machine_tol = (
            np.finfo(np.float64).eps
            * max(whitened_cross.shape)
            * max(float(eig.max()) if eig.size else 0.0, 1.0)
        )
        eig_tol = max(self.eigenvalue_tol, machine_tol)
        nonzero = int(np.count_nonzero(eig > eig_tol))
        available = min(nonzero, rank_x, rank_y, n - 1, len(corr))
        if available < 1:
            raise ValueError("CCA produced no non-zero canonical components")

        p = p[:, :available]
        q = qt.T[:, :available]
        corr = corr[:available]
        eig = eig[:available]

        invsqrt_x = 1.0 / np.sqrt(lam_x + self.ridge)
        invsqrt_y = 1.0 / np.sqrt(lam_y + self.ridge)
        self.wx = vtx.T @ (invsqrt_x[:, None] * p)
        self.wy = vty.T @ (invsqrt_y[:, None] * q)
        self.corr_ = corr.astype(np.float64)
        self.eig_ = eig.astype(np.float64)
        self.summary_ = RCCASummary(
            ridge=self.ridge,
            rank_x=int(rank_x),
            rank_y=int(rank_y),
            nonzero_components=int(nonzero),
            available_components=int(available),
            eigenvalue_tolerance=float(eig_tol),
            mean_train_corr=float(np.mean(corr)),
            min_train_corr=float(np.min(corr)),
            max_train_corr=float(np.max(corr)),
            corr_ge_0999=int(np.sum(corr >= 0.999)),
        )
        return self

    @property
    def available_components(self) -> int:
        if self.summary_ is None:
            raise RuntimeError("CCA not fitted")
        return int(self.summary_.available_components)

    def transform_components(
        self, x: np.ndarray, y: np.ndarray, n_components: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.mean_x is None or self.mean_y is None or self.wx is None or self.wy is None:
            raise RuntimeError("CCA not fitted")
        xc = np.asarray(x, dtype=np.float64) - self.mean_x
        yc = np.asarray(y, dtype=np.float64) - self.mean_y
        k = self.available_components if n_components is None else min(int(n_components), self.available_components)
        if k < 1:
            raise ValueError("Resolved CCA component count <1")
        return (
            (xc @ self.wx[:, :k]).astype(np.float32),
            (yc @ self.wy[:, :k]).astype(np.float32),
        )

    def metadata(self) -> Dict[str, Any]:
        if self.summary_ is None or self.corr_ is None or self.eig_ is None:
            raise RuntimeError("CCA not fitted")
        return {
            **asdict(self.summary_),
            "canonical_correlations": self.corr_.tolist(),
            "canonical_eigenvalues": self.eig_.tolist(),
        }


class ViewReducer:
    def __init__(self, pca_components: int, standardize: bool, seed: int):
        self.requested = int(pca_components)
        self.standardize = bool(standardize)
        self.seed = int(seed)
        self.scaler: Optional[StandardScaler] = None
        self.pca: Optional[PCA] = None
        self.used_: int = 0
        self.evr_: float = 1.0

    def fit(self, x: np.ndarray) -> "ViewReducer":
        z = np.asarray(x, dtype=np.float64)

        if self.standardize:
            self.scaler = StandardScaler(with_mean=True, with_std=True)
            z = self.scaler.fit_transform(z)

        if self.requested > 0:
            max_supported = min(z.shape[0] - 1, z.shape[1])
            k = min(self.requested, max_supported)
            if k < 1:
                raise ValueError("PCA resolved to zero components")
            self.pca = PCA(n_components=k, svd_solver="full")
            self.pca.fit(z)
            self.used_ = int(k)
            self.evr_ = float(np.sum(self.pca.explained_variance_ratio_))
        else:
            self.pca = None
            self.used_ = int(z.shape[1])
            self.evr_ = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        z = np.asarray(x, dtype=np.float64)
        if self.scaler is not None:
            z = self.scaler.transform(z)
        if self.pca is not None:
            z = self.pca.transform(z)
        return z.astype(np.float64)

    def metadata(self) -> Dict[str, Any]:
        return {
            "pca_components_requested": self.requested,
            "pca_components_used": self.used_,
            "standardize": self.standardize,
            "pca_explained_variance_ratio_sum": self.evr_,
        }


class IndependentPCARCCA:
    """Separate PCA dimensionality for FC-3 (x) and FC-4 (y)."""

    def __init__(
        self,
        *,
        pca_x: int,
        pca_y: int,
        standardize: bool,
        ridge: float,
        seed: int,
        eigenvalue_tol: float = 1e-12,
    ):
        self.rx = ViewReducer(pca_x, standardize, seed)
        self.ry = ViewReducer(pca_y, standardize, seed)
        self.cca = RidgeCCA(ridge=ridge, eigenvalue_tol=eigenvalue_tol)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "IndependentPCARCCA":
        self.rx.fit(x)
        self.ry.fit(y)
        xr = self.rx.transform(x)
        yr = self.ry.transform(y)
        self.cca.fit(xr, yr)
        return self

    @property
    def available_components(self) -> int:
        return self.cca.available_components

    def transform_components(
        self, x: np.ndarray, y: np.ndarray, n_components: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        xr = self.rx.transform(x)
        yr = self.ry.transform(y)
        return self.cca.transform_components(xr, yr, n_components)

    def transform_fused(
        self, x: np.ndarray, y: np.ndarray, n_components: int, fusion: str
    ) -> np.ndarray:
        zx, zy = self.transform_components(x, y, n_components)
        if fusion == "concat":
            return np.concatenate([zx, zy], axis=1).astype(np.float32)
        if fusion == "sum":
            return (zx + zy).astype(np.float32)
        raise ValueError("fusion must be concat or sum")

    def metadata(self) -> Dict[str, Any]:
        return {
            "x_reducer": self.rx.metadata(),
            "y_reducer": self.ry.metadata(),
            "cca": self.cca.metadata(),
        }
