
"""
Channel estimator architectures for residual / non-residual comparison.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .config import ModelConfig


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        use_bn: bool = True,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=not use_bn,
            )
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))

        if activation == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif activation == "leaky_relu":
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif activation == "gelu":
            layers.append(nn.GELU())
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))

        self.block = nn.Sequential(*layers)

    @property
    def conv(self) -> nn.Conv2d:
        return self.block[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BaseEstimator(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.use_residual = config.architecture == "residual"
        self.residual_scale = config.residual_scale

    def _finalize_output(self, x: torch.Tensor, predicted_map: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.residual_scale * predicted_map
        return predicted_map

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)


class StandardCNN(BaseEstimator):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        c = config.input_shape[0]
        nf = config.num_filters
        kwargs = dict(use_bn=config.use_batch_norm, activation=config.activation, dropout=config.dropout)

        self.enc1 = ConvBlock(c, nf, 9, **kwargs)
        self.enc2 = ConvBlock(nf, nf * 2, 5, **kwargs)
        self.enc3 = ConvBlock(nf * 2, nf * 4, 3, **kwargs)

        self.bottleneck = nn.Sequential(
            ConvBlock(nf * 4, nf * 4, 3, **kwargs),
            ConvBlock(nf * 4, nf * 4, 3, **kwargs),
        )

        self.dec3 = ConvBlock(nf * 4, nf * 2, 3, **kwargs)
        self.dec2 = ConvBlock(nf * 2, nf, 5, **kwargs)
        self.dec1 = ConvBlock(nf, nf, 9, **kwargs)
        self.output_head = nn.Conv2d(nf, c, 1)

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.bottleneck(e3)
        d3 = self.dec3(b)
        d2 = self.dec2(d3)
        d1 = self.dec1(d2)
        predicted = self.output_head(d1)
        return self._finalize_output(x, predicted)


class DeepEncoderCNN(BaseEstimator):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        c = config.input_shape[0]
        nf = config.num_filters
        kwargs = dict(use_bn=config.use_batch_norm, activation=config.activation, dropout=config.dropout)

        self.init_conv = ConvBlock(c, nf, 7, **kwargs)
        self.enc1 = nn.Sequential(ConvBlock(nf, nf * 2, 3, **kwargs), ConvBlock(nf * 2, nf * 2, 3, **kwargs))
        self.enc2 = nn.Sequential(ConvBlock(nf * 2, nf * 4, 3, **kwargs), ConvBlock(nf * 4, nf * 4, 3, **kwargs))
        self.enc3 = nn.Sequential(ConvBlock(nf * 4, nf * 8, 3, **kwargs), ConvBlock(nf * 8, nf * 8, 3, **kwargs))
        self.middle = nn.Sequential(
            ConvBlock(nf * 8, nf * 8, 3, **kwargs),
            ConvBlock(nf * 8, nf * 8, 3, **kwargs),
            ConvBlock(nf * 8, nf * 8, 3, **kwargs),
        )
        self.dec3 = nn.Sequential(ConvBlock(nf * 8, nf * 4, 3, **kwargs), ConvBlock(nf * 4, nf * 4, 3, **kwargs))
        self.dec2 = nn.Sequential(ConvBlock(nf * 4, nf * 2, 3, **kwargs), ConvBlock(nf * 2, nf * 2, 3, **kwargs))
        self.dec1 = nn.Sequential(ConvBlock(nf * 2, nf, 3, **kwargs), ConvBlock(nf, nf, 3, **kwargs))
        self.output_conv = nn.Conv2d(nf, c, 1)

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.init_conv(x)
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        xm = self.middle(x3)
        d3 = self.dec3(xm)
        d2 = self.dec2(d3)
        d1 = self.dec1(d2)
        predicted = self.output_conv(d1)
        return self._finalize_output(x, predicted)


class UNetStyleCNN(BaseEstimator):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        c = config.input_shape[0]
        nf = config.num_filters
        kwargs = dict(use_bn=config.use_batch_norm, activation=config.activation, dropout=config.dropout)

        self.enc1 = ConvBlock(c, nf, 7, **kwargs)
        self.enc2 = ConvBlock(nf, nf * 2, 5, **kwargs)
        self.enc3 = ConvBlock(nf * 2, nf * 4, 3, **kwargs)
        self.middle = ConvBlock(nf * 4, nf * 8, 3, **kwargs)
        self.dec3 = ConvBlock(nf * 8 + nf * 4, nf * 4, 3, **kwargs)
        self.dec2 = ConvBlock(nf * 4 + nf * 2, nf * 2, 5, **kwargs)
        self.dec1 = ConvBlock(nf * 2 + nf, nf, 7, **kwargs)
        self.output_conv = nn.Conv2d(nf, c, 1)

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        m = self.middle(e3)
        d3 = self.dec3(torch.cat([m, e3], dim=1))
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))
        predicted = self.output_conv(d1)
        return self._finalize_output(x, predicted)


class SimpleCNN(BaseEstimator):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        c = config.input_shape[0]
        nf = config.num_filters
        kwargs = dict(use_bn=config.use_batch_norm, activation=config.activation, dropout=config.dropout)
        self.features = nn.Sequential(
            ConvBlock(c, nf, 9, **kwargs),
            ConvBlock(nf, nf * 2, 5, **kwargs),
            ConvBlock(nf * 2, nf * 2, 3, **kwargs),
            ConvBlock(nf * 2, nf, 3, **kwargs),
        )
        self.output_conv = nn.Conv2d(nf, c, 1)
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        predicted = self.output_conv(self.features(x))
        return self._finalize_output(x, predicted)


def create_model(config: ModelConfig, model_type: Optional[str] = None) -> nn.Module:
    model_key = model_type or config.model_variant
    models = {
        "standard": StandardCNN,
        "deep": DeepEncoderCNN,
        "unet": UNetStyleCNN,
        "simple": SimpleCNN,
    }
    if model_key not in models:
        raise ValueError(f"Unknown model variant: {model_key}")
    return models[model_key](config)


def _resolve_hook_target(base_model: nn.Module) -> Dict[str, nn.Module]:
    candidates: Dict[str, nn.Module] = {}
    for name, module in base_model.named_modules():
        if isinstance(module, ConvBlock):
            candidates[name] = module.conv
        elif isinstance(module, nn.Conv2d) and "output" not in name and "head" not in name:
            candidates.setdefault(name, module)
    if not candidates:
        raise RuntimeError("No convolutional modules found for activation hooks.")
    selected: Dict[str, nn.Module] = {}
    for idx, (name, module) in enumerate(candidates.items()):
        selected[name] = module
        if idx >= 5:
            break
    return selected


class ChannelEstimatorWithHooks(nn.Module):
    """Generic hook-enabled wrapper around any estimator variant."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.base_model = create_model(config)
        self.activation_maps: Dict[str, torch.Tensor] = {}
        self.hook_modules = _resolve_hook_target(self.base_model)
        self._hook_handles = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        for name, module in self.hook_modules.items():
            handle = module.register_forward_hook(self._build_hook(name))
            self._hook_handles.append(handle)

    def _build_hook(self, name: str):
        def hook(_module, _inputs, output):
            self.activation_maps[name] = output.detach()
        return hook

    def clear_activations(self) -> None:
        self.activation_maps.clear()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model(x)

    def load_plain_state_dict(self, state_dict) -> None:
        self.base_model.load_state_dict(state_dict)

    def state_dict(self, *args, **kwargs):  # type: ignore[override]
        return self.base_model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True):  # type: ignore[override]
        return self.base_model.load_state_dict(state_dict, strict=strict)


class ChannelEstimationLoss(nn.Module):
    def __init__(self, mse_weight: float = 1.0, l1_weight: float = 0.0):
        super().__init__()
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.mse_weight * self.mse_loss(prediction, target)
        if self.l1_weight > 0:
            loss = loss + self.l1_weight * self.l1_loss(prediction, target)
        return loss


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model: nn.Module, input_shape: Tuple[int, ...]) -> str:
    summary = []
    config = getattr(model, "config", getattr(model, "base_model", None))
    architecture = getattr(config, "architecture", "unknown")
    summary.append("=" * 60)
    summary.append(f"MODEL SUMMARY ({architecture})")
    summary.append("=" * 60)
    summary.append(f"Input shape: {input_shape}")
    total_params = count_parameters(model)
    summary.append(f"Trainable parameters: {total_params:,}")
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
            params = sum(p.numel() for p in module.parameters())
            summary.append(f"  {name}: {module.__class__.__name__} ({params:,} params)")
    return "\n".join(summary)


# Backward-compatible name used in the original project.
ChannelEstimator = StandardCNN


if __name__ == "__main__":
    cfg = ModelConfig()
    for arch in ["non_residual", "residual"]:
        for variant in ["standard", "simple"]:
            cfg.architecture = arch
            cfg.model_variant = variant
            model = create_model(cfg)
            x = torch.randn(2, *cfg.input_shape)
            y = model(x)
            print(arch, variant, y.shape, count_parameters(model))
