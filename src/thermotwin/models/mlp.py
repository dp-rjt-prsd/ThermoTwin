"""Small feed-forward model for normalized one-step cell-temperature deltas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

from thermotwin.models.config import MLPConfig


@dataclass(frozen=True, slots=True)
class MLPArchitectureMetadata:
    """Immutable, JSON-serializable description of an MLP architecture."""

    input_dim: int
    output_dim: int
    hidden_widths: tuple[int, ...]
    activation: str


class ThermalMLP(nn.Module):
    """Map normalized thermal features to normalized cell-temperature deltas."""

    def __init__(self, config: MLPConfig) -> None:
        super().__init__()
        if not isinstance(config, MLPConfig):
            raise TypeError("config must be an MLPConfig")
        self._architecture_metadata = MLPArchitectureMetadata(
            input_dim=config.input_dim,
            output_dim=config.output_dim,
            hidden_widths=config.hidden_widths,
            activation=config.activation,
        )

        layers: list[nn.Module] = []
        dimensions = (config.input_dim, *config.hidden_widths, config.output_dim)
        for index, (input_dim, output_dim) in enumerate(
            zip(dimensions, dimensions[1:], strict=False)
        ):
            layers.append(nn.Linear(input_dim, output_dim))
            if index < len(config.hidden_widths):
                layers.append(nn.GELU())
        self.network = nn.Sequential(*layers)

    @property
    def architecture_metadata(self) -> MLPArchitectureMetadata:
        """Return the immutable description needed to reconstruct this model."""
        return self._architecture_metadata

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Validate finite floating features and predict normalized deltas."""
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("inputs must be a torch.Tensor")
        if inputs.ndim != 2:
            raise ValueError("inputs must have rank 2 [batch, input_dim]")
        if inputs.shape[0] == 0:
            raise ValueError("inputs batch must be nonempty")
        if inputs.shape[1] != self._architecture_metadata.input_dim:
            raise ValueError(
                f"inputs must have input_dim {self._architecture_metadata.input_dim}"
            )
        if not inputs.is_floating_point():
            raise TypeError("inputs must have a floating dtype")
        if not torch.isfinite(inputs).all().item():
            raise ValueError("inputs must contain only finite values")
        return cast(torch.Tensor, self.network(inputs))
