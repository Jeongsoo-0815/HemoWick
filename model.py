"""HemoWick image backbones, patch selection and temporal Hb/Hct regression."""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from config import AppConfig, ModelConfig, RESNET_BACKBONES, VIT_BACKBONES


class PatchPruningBlock(nn.Module):
    """Prune patch tokens before the wrapped torchvision encoder block."""

    def __init__(
        self,
        original_block: nn.Module,
        pruning_ratio: float,
        pruning_method: str,
    ):
        super().__init__()
        self.original_block = original_block
        self.pruning_ratio = float(pruning_ratio)
        self.pruning_method = pruning_method
        self.last_keep_indices: torch.Tensor | None = None

    def _indices(self, cls_token: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        batch_size, patch_count, _ = patches.shape
        keep_count = max(1, min(patch_count, int((1.0 - self.pruning_ratio) * patch_count)))
        if keep_count == patch_count:
            return torch.arange(patch_count, device=patches.device).expand(batch_size, -1)

        if self.pruning_method == "uniform":
            positions = torch.linspace(
                0, patch_count - 1, keep_count, device=patches.device
            ).round().long()
            return positions.expand(batch_size, -1)

        cls_normalized = F.normalize(cls_token, p=2, dim=-1)
        patch_normalized = F.normalize(patches, p=2, dim=-1)
        importance = torch.sum(patch_normalized * cls_normalized, dim=-1)
        indices = torch.topk(importance, keep_count, dim=1).indices
        return torch.sort(indices, dim=1).values

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.pruning_ratio <= 0.0 or tokens.shape[1] <= 2:
            self.last_keep_indices = torch.arange(
                max(0, tokens.shape[1] - 1), device=tokens.device
            ).expand(tokens.shape[0], -1)
            return self.original_block(tokens)

        cls_token = tokens[:, :1]
        patches = tokens[:, 1:]
        indices = self._indices(cls_token, patches)
        gather_index = indices.unsqueeze(-1).expand(-1, -1, patches.shape[-1])
        selected = torch.gather(patches, dim=1, index=gather_index)
        self.last_keep_indices = indices.detach()
        return self.original_block(torch.cat([cls_token, selected], dim=1))


class DynamicViTBackbone(nn.Module):
    """ImageNet-pretrained ViT with optional adaptive or uniform patch pruning."""

    def __init__(
        self,
        name: str = "vit_b_16",
        pruning_ratio: float = 0.0,
        pruning_method: str = "adaptive",
        pretrained: bool = True,
    ):
        super().__init__()
        if name == "vit_b_16":
            weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
            vit = models.vit_b_16(weights=weights)
        elif name == "vit_b_32":
            weights = models.ViT_B_32_Weights.IMAGENET1K_V1 if pretrained else None
            vit = models.vit_b_32(weights=weights)
        else:
            raise ValueError(f"Unsupported ViT backbone: {name}")

        self.patch_embed = vit.conv_proj
        self.class_token = vit.class_token
        self.position_embedding = vit.encoder.pos_embedding
        self.encoder_dropout = vit.encoder.dropout
        self.encoder_norm = vit.encoder.ln
        self.embed_dim = int(vit.hidden_dim)
        self.pruning_ratio = float(pruning_ratio)
        self.pruning_method = pruning_method

        blocks: list[nn.Module] = []
        for index, block in enumerate(vit.encoder.layers):
            if index == 0:
                blocks.append(PatchPruningBlock(block, pruning_ratio, pruning_method))
            else:
                blocks.append(block)
        self.encoder_blocks = nn.ModuleList(blocks)

    def _position_embedding(self, height: int, width: int) -> torch.Tensor:
        patch_positions = self.position_embedding[:, 1:]
        original_side = int(math.sqrt(patch_positions.shape[1]))
        if original_side * original_side != patch_positions.shape[1]:
            raise RuntimeError("Expected a square ViT positional-embedding grid.")

        if original_side == height and original_side == width:
            return self.position_embedding

        patch_positions = patch_positions.reshape(
            1, original_side, original_side, self.embed_dim
        ).permute(0, 3, 1, 2)
        patch_positions = F.interpolate(
            patch_positions,
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        )
        patch_positions = patch_positions.permute(0, 2, 3, 1).reshape(
            1, height * width, self.embed_dim
        )
        return torch.cat([self.position_embedding[:, :1], patch_positions], dim=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embed(images)
        batch_size, _, height, width = patches.shape
        tokens = patches.flatten(2).transpose(1, 2)
        cls = self.class_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self._position_embedding(height, width)
        tokens = self.encoder_dropout(tokens)
        for block in self.encoder_blocks:
            tokens = block(tokens)
        tokens = self.encoder_norm(tokens)
        return tokens[:, 0]

    @property
    def last_keep_indices(self) -> torch.Tensor | None:
        first_block = self.encoder_blocks[0]
        if isinstance(first_block, PatchPruningBlock):
            return first_block.last_keep_indices
        return None


class ResNetBackbone(nn.Module):
    def __init__(self, name: str, pretrained: bool):
        super().__init__()
        builders = {
            "resnet18": (models.resnet18, models.ResNet18_Weights),
            "resnet34": (models.resnet34, models.ResNet34_Weights),
            "resnet50": (models.resnet50, models.ResNet50_Weights),
            "resnet101": (models.resnet101, models.ResNet101_Weights),
        }
        builder, weight_enum = builders[name]
        weights = weight_enum.IMAGENET1K_V1 if pretrained else None
        network = builder(weights=weights)
        self.embed_dim = int(network.fc.in_features)
        network.fc = nn.Identity()
        self.network = network

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)


def build_backbone(config: ModelConfig) -> nn.Module:
    if config.backbone in VIT_BACKBONES:
        return DynamicViTBackbone(
            name=config.backbone,
            pruning_ratio=config.pruning_ratio,
            pruning_method=config.pruning_method,
            pretrained=config.pretrained,
        )
    if config.backbone in RESNET_BACKBONES:
        return ResNetBackbone(config.backbone, pretrained=config.pretrained)
    raise ValueError(f"Unsupported backbone: {config.backbone}")


class HemoWickRegressor(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.backbone = build_backbone(config)
        feature_dim = int(self.backbone.embed_dim)

        self.temporal_mode = config.temporal_mode
        temporal_dim = feature_dim
        if self.temporal_mode in {"lstm", "gru"}:
            recurrent_class = nn.LSTM if self.temporal_mode == "lstm" else nn.GRU
            self.temporal = recurrent_class(
                input_size=feature_dim,
                hidden_size=config.rnn_hidden_size,
                num_layers=config.rnn_num_layers,
                batch_first=True,
                bidirectional=config.rnn_bidirectional,
            )
            temporal_dim = config.rnn_hidden_size * (
                2 if config.rnn_bidirectional else 1
            )
        else:
            self.temporal = None

        self.regressor = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(temporal_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(128, 2),
        )

    def _temporal_features(self, features: torch.Tensor) -> torch.Tensor:
        if self.temporal_mode == "single":
            if features.shape[1] != 1:
                raise ValueError("Single-image mode expects one frame per sample.")
            return features[:, 0]
        if self.temporal_mode == "mean":
            return features.mean(dim=1)

        if self.temporal is None:
            raise RuntimeError("Temporal module was not initialized.")
        _, hidden = self.temporal(features)
        if self.temporal_mode == "lstm":
            hidden = hidden[0]
        if self.config.rnn_bidirectional:
            return torch.cat([hidden[-2], hidden[-1]], dim=1)
        return hidden[-1]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 5:
            raise ValueError("Expected images with shape [batch, time, channels, height, width].")
        batch_size, time_steps, channels, height, width = images.shape
        flattened = images.reshape(batch_size * time_steps, channels, height, width)
        features = self.backbone(flattened).reshape(batch_size, time_steps, -1)
        return self.regressor(self._temporal_features(features))


def build_model(config: AppConfig) -> HemoWickRegressor:
    return HemoWickRegressor(config.model)
