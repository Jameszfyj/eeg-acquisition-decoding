"""从参考工程提取的基础双频带IFNetV2；保留来源，去除timm和实验模型依赖。"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

class Conv1dWithConstraint(nn.Conv1d):
    def __init__(self, *args, max_norm: float = 1.0, do_weight_norm: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_norm = float(max_norm)
        self.do_weight_norm = bool(do_weight_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.do_weight_norm:
            with torch.no_grad():
                self.weight.copy_(torch.renorm(self.weight, p=2, dim=0, maxnorm=self.max_norm))
        return super().forward(x)

class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, max_norm: float = 1.0, do_weight_norm: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_norm = float(max_norm)
        self.do_weight_norm = bool(do_weight_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.do_weight_norm:
            with torch.no_grad():
                self.weight.copy_(torch.renorm(self.weight, p=2, dim=0, maxnorm=self.max_norm))
        return super().forward(x)

class LogPowerLayer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.clamp(torch.mean(x.square(), dim=self.dim), 1e-4, 1e4))

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

class IFStem(nn.Module):
    def __init__(
        self,
        in_planes: int,
        out_planes: int = 64,
        kernel_size: int = 63,
        patch_size: int = 125,
        radix: int = 2,
    ) -> None:
        super().__init__()
        self.out_planes = int(out_planes)
        self.radix = int(radix)
        self.patch_size = int(patch_size)
        self.temporal = IFStemTemporal(
            in_planes=in_planes,
            out_planes=out_planes,
            kernel_size=kernel_size,
            radix=radix,
        )
        self.dropout = nn.Dropout(0.5)
        self.log_power = LogPowerLayer(dim=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, n_times = x.shape
        x = self.temporal(x)
        usable = (n_times // self.patch_size) * self.patch_size
        if usable <= 0:
            raise ValueError("IFNet patch size is larger than the available time points.")
        x = x[..., :usable]
        x = x.reshape(batch_size, self.out_planes, usable // self.patch_size, self.patch_size)
        x = self.log_power(x)
        return self.dropout(x)

class IFNetClassifier(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_times: int,
        n_classes: int,
        radix: int = 2,
        out_planes: int = 64,
        kernel_size: int = 63,
        patch_size: int = 125,
    ) -> None:
        super().__init__()
        self.radix = int(radix)
        self.n_channels = int(n_channels)
        self.stem = IFStem(
            in_planes=n_channels * radix,
            out_planes=out_planes,
            kernel_size=kernel_size,
            patch_size=patch_size,
            radix=radix,
        )
        n_patches = max(1, n_times // patch_size)
        self.classifier = nn.Sequential(
            LinearWithConstraint(out_planes * n_patches, n_classes, max_norm=0.5),
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            batch_size, bands, channels, n_times = x.shape
            x = x.reshape(batch_size, bands * channels, n_times)
        x = self.stem(x)
        x = x.flatten(1)
        return self.classifier(x)
