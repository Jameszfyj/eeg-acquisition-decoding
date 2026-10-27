"""Inference-only IFNetRTG-L definitions from the supplied cargame source.

Definitions retain the original operations and state-dict keys. Full training
protocols remain in the upstream training environment; see NOTICE.md.
"""
from __future__ import annotations
from typing import Optional, Sequence, Tuple
import numpy as np
from scipy import signal
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_



class TemporalPowerPool(nn.Module):
    """Weighted log-power pooling over time: log(mean(weights * x^2)) -> (B, C)."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        if weights.dim() == 1:
            weights = weights.unsqueeze(0)
        while weights.dim() < x.dim():
            weights = weights.unsqueeze(1)
        denom = weights.sum(dim=-1).clamp_min(self.eps)
        power = (x * x) * weights
        pooled = power.sum(dim=-1) / denom
        return torch.log(pooled + self.eps)

def uniform_weights(n_times: int, device: Optional[torch.device] = None) -> torch.Tensor:
    return torch.ones(n_times, device=device)

class Conv1dWithConstraint(nn.Conv1d):
    def __init__(self, *args, max_norm: float = 1.0, do_weight_norm: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_norm = float(max_norm)
        self.do_weight_norm = bool(do_weight_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.do_weight_norm:
            with torch.no_grad():
                self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)

class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, max_norm: float = 1.0, do_weight_norm: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_norm = float(max_norm)
        self.do_weight_norm = bool(do_weight_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.do_weight_norm:
            with torch.no_grad():
                self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)

class IFStemTemporal(nn.Module):
    """IFNet stem without patchify/log-power: returns temporal features (B, C, n_times).

    zero_padding=False: the subband temporal convs use VALID convolution (no
    zero padding); each branch output is cropped to the shortest branch length
    (T - max_kernel + 1) before summing, so the conv edge region is never
    computed instead of being zero-padded.
    """

    def __init__(
        self,
        in_planes: int,
        out_planes: int = 64,
        kernel_size: int = 63,
        radix: int = 2,
        zero_padding: bool = True,
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()
        self.out_planes = int(out_planes)
        self.radix = int(radix)
        self.zero_padding = bool(zero_padding)
        self.padding_mode = str(padding_mode)
        self.spatial = nn.Sequential(
            Conv1dWithConstraint(
                in_planes,
                out_planes * radix,
                kernel_size=1,
                groups=radix,
                bias=False,
                max_norm=2.0,
                do_weight_norm=False,
            ),
            nn.BatchNorm1d(out_planes * radix),
        )
        branches = []
        branch_kernel = int(kernel_size)
        self.max_kernel = branch_kernel
        for _ in range(radix):
            branches.append(
                nn.Sequential(
                    nn.Conv1d(
                        out_planes,
                        out_planes,
                        kernel_size=branch_kernel,
                        stride=1,
                        groups=out_planes,
                        padding=branch_kernel // 2 if self.zero_padding else 0,
                        padding_mode=self.padding_mode if self.zero_padding else "zeros",
                        bias=False,
                    ),
                    nn.BatchNorm1d(out_planes),
                )
            )
            branch_kernel = max(3, branch_kernel // 2)
        self.temporal_branches = nn.ModuleList(branches)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial(x)
        chunks = torch.chunk(x, self.radix, dim=1)
        outputs = [branch(chunk) for branch, chunk in zip(self.temporal_branches, chunks)]
        if not self.zero_padding:
            crop = x.shape[-1] - self.max_kernel + 1
            if crop <= 0:
                raise ValueError(
                    f"IFStemTemporal valid conv requires n_times > max_kernel "
                    f"({x.shape[-1]} <= {self.max_kernel})."
                )
            outputs = [branch_out[..., :crop] for branch_out in outputs]
        x = sum(outputs)
        return F.gelu(x)

class IFNetGClassifier(nn.Module):
    """IFNet with global pooling over the full time series.

    Input: (B, bands, n_channels, n_times) with arbitrary n_times.
    zero_padding=False: the subband temporal convs are VALID (no zero padding),
    so the conv edge region is never computed/pooled.
    pool_mode:
      "power"       : log(mean_t x^2) over the whole window (log of average var)
      "logvar_mean" : mean over patches of log(mean_patch x^2)
                      (average of logvars - the "0.5 s version" of global)
    pool_patch_size / pool_patch_stride: patch geometry for "logvar_mean";
    stride 0 = non-overlapping patches, stride > 0 = sliding patches.
    pool_tail_policy:
      "drop"       : preserve the legacy behavior and ignore an incomplete tail.
      "cover_last" : append one end-aligned patch when the regular patch grid
                     does not cover the final feature samples.
    input_norm: per-window, per-channel (minus mean) / (div std) z-score applied
                to the model input right before the first spatial filter, so the
                convs see correlation-like normalized signals with no influence
                of absolute variance power.
    multiscale: multiscale logvar pooling over pool_scale_sizes patches at a
                fixed pool_patch_stride (the "average dB" strategy set).
                Training: one scale is randomly selected per forward (batch).
                Eval: every scale is pooled and the features are averaged.
    """

    def __init__(
        self,
        n_channels: int,
        n_times: int,
        n_classes: int,
        radix: int = 2,
        out_planes: int = 64,
        kernel_size: int = 63,
        zero_padding: bool = True,
        pool_mode: str = "power",
        pool_patch_size: int = 125,
        pool_patch_stride: int = 0,
        pool_tail_policy: str = "drop",
        pool_interior_offset: int = 0,
        padding_mode: str = "zeros",
        train_crop_patches: int = 0,
        input_norm: bool = False,
        multiscale: bool = False,
        pool_scale_sizes: Sequence[int] = (125, 250, 375, 500),
    ) -> None:
        super().__init__()
        self.radix = int(radix)
        self.out_planes = int(out_planes)
        self.n_channels = int(n_channels)
        self.n_times = int(n_times)
        self.zero_padding = bool(zero_padding)
        self.pool_mode = str(pool_mode)
        self.pool_patch_size = int(pool_patch_size)
        self.pool_patch_stride = int(pool_patch_stride)
        self.pool_tail_policy = str(pool_tail_policy)
        self.pool_interior_offset = int(pool_interior_offset)
        self.padding_mode = str(padding_mode)
        self.train_crop_patches = int(train_crop_patches)
        self.input_norm = bool(input_norm)
        self.multiscale = bool(multiscale)
        self.pool_scale_sizes = tuple(int(size) for size in pool_scale_sizes)
        if self.pool_tail_policy not in ("drop", "cover_last"):
            raise ValueError(
                "IFNetGClassifier pool_tail_policy must be 'drop' or 'cover_last', "
                f"got {self.pool_tail_policy!r}."
            )
        self.stem_temporal = IFStemTemporal(
            in_planes=n_channels * radix,
            out_planes=out_planes,
            kernel_size=kernel_size,
            radix=radix,
            zero_padding=self.zero_padding,
            padding_mode=self.padding_mode,
        )
        self.pool = TemporalPowerPool()
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Sequential(
            LinearWithConstraint(out_planes, n_classes, max_norm=0.5),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _pool_patch_starts(
        self,
        feature_length: int,
        patch_size: int | None = None,
        stride: int | None = None,
    ) -> tuple[int, ...]:
        patch_size = int(self.pool_patch_size if patch_size is None else patch_size)
        stride = int(self.pool_patch_stride if stride is None else stride)
        if stride <= 0:
            stride = patch_size
        offset = int(getattr(self, "pool_interior_offset", 0) or 0)
        usable = feature_length - 2 * offset  # valid interior length (full receptive field)
        if offset == 0 and feature_length < patch_size:
            raise ValueError("IFNetGClassifier logvar_mean pooling: window shorter than pool_patch_size.")
        if usable <= 0:
            raise ValueError(
                "IFNetGClassifier logvar_mean pooling: interior shorter than 2*pool_interior_offset."
            )
        patch_size = min(patch_size, usable)  # adaptive shrink for short interiors
        hi = feature_length - offset
        starts = list(range(offset, hi - patch_size + 1, stride))
        if self.pool_tail_policy == "cover_last":
            tail_start = hi - patch_size
            if starts[-1] != tail_start:
                starts.append(tail_start)
        return tuple(starts)

    def _logvar_pool(
        self,
        features: torch.Tensor,
        patch_size: int | None = None,
        patch_stride: int | None = None,
    ) -> torch.Tensor:
        feature_length = int(features.shape[-1])
        offset = int(getattr(self, "pool_interior_offset", 0) or 0)
        usable = feature_length - 2 * offset
        patch_size = min(
            int(self.pool_patch_size if patch_size is None else patch_size),
            max(1, usable),
            feature_length,
        )
        starts = self._pool_patch_starts(
            feature_length,
            patch_size=patch_size,
            stride=self.pool_patch_stride if patch_stride is None else patch_stride,
        )
        patches = torch.stack(
            [features[..., start : start + patch_size] for start in starts],
            dim=-2,
        )
        patch_logvar = torch.log(torch.clamp(patches.square().mean(dim=-1), 1e-4, 1e4))
        return patch_logvar.mean(dim=-1)

    def _logvar_pool_ms(self, features: torch.Tensor) -> torch.Tensor:
        if self.training:
            index = int(torch.randint(len(self.pool_scale_sizes), (1,)).item())
            return self._logvar_pool(features, patch_size=self.pool_scale_sizes[index])
        pooled = torch.stack(
            [self._logvar_pool(features, patch_size=size) for size in self.pool_scale_sizes],
            dim=0,
        )
        return pooled.mean(dim=0)

    def _train_crop(self, features: torch.Tensor) -> torch.Tensor:
        """Hidden random cropping (HRC): during training only, crop the stem
        features to train_crop_patches * 125 samples at a uniform offset
        d in [0, 125] (one 0.5 s slice removed from a random edge split), so
        the model sees every possible slice-grid phase of the window."""
        n = int(features.shape[-1])
        keep = int(self.train_crop_patches) * 125
        if keep <= 0 or keep >= n or not self.training:
            return features
        hi = n - keep
        d = int(torch.randint(0, hi + 1, (1,)).item())
        return features[..., d : d + keep]

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the pooled 64-D feature before dropout/classification."""
        if x.ndim == 4:
            batch_size, bands, channels, n_times = x.shape
            x = x.reshape(batch_size, bands * channels, n_times)
        if self.input_norm:
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True)
            x = (x - mean) / (std + 1e-8)
        features = self.stem_temporal(x)
        features = self._train_crop(features)
        if self.multiscale:
            return self._logvar_pool_ms(features)
        if self.pool_mode == "logvar_mean":
            return self._logvar_pool(features)
        weights = uniform_weights(features.shape[-1], device=features.device)
        return self.pool(features, weights)

    def forward(self, x: torch.Tensor, global_mode: bool = True) -> torch.Tensor:
        del global_mode
        pooled = self.forward_features(x)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)

def bandpass_array(data: np.ndarray, fs: float, band: Tuple[float, float]) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 0 or data.shape[-1] <= 0:
        return np.zeros_like(data, dtype=np.float32)
    low, high = band
    high = min(high, fs / 2.0 - 1e-3)
    if low <= 0:
        sos = signal.butter(4, high, btype="lowpass", fs=fs, output="sos")
    else:
        sos = signal.butter(4, [low, high], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, data, axis=-1).astype(np.float32)

def band_split_array(x, fs: float, bands: Sequence[Sequence[float]], mode: str = "concat"):
    """THE band split of the IFNet family: one zero-phase bandpass per band.

    ``mode="concat"`` returns the band-major channel concatenation ``(N, B*C, T)``
    - the layout IFNetG / IFNetCombine were trained on; ``mode="stack"`` returns
    ``(N, B, C, T)``, the layout the older IFNet / FBCNet protocols use. Both are
    built from the same ``bandpass_array`` the training pipeline has always used,
    so the stage applied here and a dataset pre-split with the same bands are the
    SAME numbers - that is what makes an already-split training run and a raw
    inference window interchangeable.
    """
    values = np.asarray(x, dtype=np.float32)

    parts = [bandpass_array(values, fs, (float(band[0]), float(band[1]))) for band in bands]
    if str(mode) == "stack":
        return np.stack(parts, axis=1).astype(np.float32)
    return np.concatenate(parts, axis=1).astype(np.float32)
