from typing import Literal

import torch
import torch.nn as nn


class TritonMSELoss(nn.Module):
    def __init__(self, reduction: Literal["none", "mean", "sum"] = "mean") -> None:
        super().__init__()
        if reduction not in ("none", "mean", "sum"):
            raise ValueError("reduction must be 'none', 'mean', or 'sum'")
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if input.shape != target.shape:
            raise ValueError(f"input and target shapes must match, got {input.shape} and {target.shape}")
        if not input.is_cuda:
            diff = input - target
            loss = diff.square()
            if self.reduction == "sum":
                return loss.sum()
            if self.reduction == "mean":
                return loss.mean()
            return loss
        from utils.triton_primitives import mse_loss
        return mse_loss(input, target, self.reduction)


class TritonCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        weight: torch.Tensor | None = None,
        ignore_index: int = -100,
        reduction: Literal["none", "mean", "sum"] = "mean",
    ) -> None:
        super().__init__()
        self.register_buffer("weight", weight)
        self.ignore_index = ignore_index
        self.reduction = reduction
        if reduction not in ("none", "mean", "sum"):
            raise ValueError("reduction must be 'none', 'mean', or 'sum'")

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if input.ndim < 2:
            raise ValueError("input must have shape (N, C, ...)")
        if not input.is_cuda:
            return nn.functional.cross_entropy(
                input,
                target,
                weight=self.weight,
                ignore_index=self.ignore_index,
                reduction=self.reduction,
            )
        from utils.triton_primitives import cross_entropy_loss
        return cross_entropy_loss(input, target, self.weight, self.ignore_index, self.reduction)


class TritonBCEWithLogitsLoss(nn.Module):
    def __init__(
        self,
        weight: torch.Tensor | None = None,
        reduction: Literal["none", "mean", "sum"] = "mean",
    ) -> None:
        super().__init__()
        self.register_buffer("weight", weight)
        self.reduction = reduction
        if reduction not in ("none", "mean", "sum"):
            raise ValueError("reduction must be 'none', 'mean', or 'sum'")

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if input.shape != target.shape:
            raise ValueError(f"input and target shapes must match, got {input.shape} and {target.shape}")
        if not input.is_cuda:
            return nn.functional.binary_cross_entropy_with_logits(
                input, target, weight=self.weight, reduction=self.reduction
            )
        from utils.triton_primitives import bce_with_logits_loss
        return bce_with_logits_loss(input, target, self.weight, self.reduction)


__all__ = ["TritonMSELoss", "TritonCrossEntropyLoss", "TritonBCEWithLogitsLoss"]
