"""MoCo v2 SSL pretraining using Lightly framework components.

Single-GPU friendly via Lightly's batch_shuffle utility (prevents BN info leak).

Based on Lightly tutorial: https://docs.lightly.ai/tutorials/package/tutorial_moco_memory_bank.html
"""
import copy
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

from lightly.loss import NTXentLoss
from lightly.models.modules.heads import MoCoProjectionHead
# from lightly.models.utils import replace_batchnorm_with_groupnorm
from lightly.models.utils import (
    batch_shuffle,
    batch_unshuffle,
    deactivate_requires_grad,
    update_momentum,
)


class MoCoBackbone(nn.Module):
    """SMP encoder + global pooling, returns flat features."""

    def __init__(self, encoder_name: str = "resnet50", weights: str = "imagenet"):
        super().__init__()
        self.encoder = smp.encoders.get_encoder(
            encoder_name, in_channels=3, depth=5, weights=weights
        )
        # replace_batchnorm_with_groupnorm(self.encoder)
        self.out_dim = self.encoder.out_channels[-1]  # 2048 for resnet50

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        x = features[-1]
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return x


class MoCo(nn.Module):
    """
    MoCo v2 using Lightly components.

    Args:
        encoder_name: backbone arch (default 'resnet50')
        proj_dim: projection dim (default 128)
        proj_hidden_dim: hidden MLP dim (default 2048)
        queue_size: memory bank size (default 65536)
        momentum: EMA coefficient (default 0.999)
        temperature: InfoNCE temperature (default 0.2)
        encoder_weights: 'imagenet' or None
    """

    def __init__(
        self,
        encoder_name: str = "resnet50",
        proj_dim: int = 128,
        proj_hidden_dim: int = 2048,
        queue_size: int = 65536,
        momentum: float = 0.999,
        temperature: float = 0.2,
        encoder_weights: str = "imagenet",
    ):
        super().__init__()
        self.momentum_value = momentum

        # Query encoder
        self.backbone = MoCoBackbone(encoder_name, weights=encoder_weights)
        feat_dim = self.backbone.out_dim
        self.projection_head = MoCoProjectionHead(feat_dim, proj_hidden_dim, proj_dim)

        # Key encoder = momentum copy
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        # Loss with memory bank (Lightly's queue equivalent)
        self.criterion = NTXentLoss(
            temperature=temperature,
            memory_bank_size=(queue_size, proj_dim),
        )

    def forward_query(self, x: torch.Tensor) -> torch.Tensor:
        q = self.backbone(x)
        q = self.projection_head(q)
        return q

    @torch.no_grad()
    def forward_key(self, x: torch.Tensor) -> torch.Tensor:
        # Batch shuffle: critical for single-GPU MoCo
        x_shuffled, shuffle_idx = batch_shuffle(x)
        k = self.backbone_momentum(x_shuffled)
        k = self.projection_head_momentum(k)
        k = batch_unshuffle(k, shuffle_idx)
        return k

    def forward(self, view_q: torch.Tensor, view_k: torch.Tensor) -> torch.Tensor:
        """Returns loss directly (NTXentLoss handles logits and labels internally)."""
        # Momentum update of key encoder
        update_momentum(self.backbone, self.backbone_momentum, m=self.momentum_value)
        update_momentum(
            self.projection_head, self.projection_head_momentum, m=self.momentum_value
        )

        q = self.forward_query(view_q)
        k = self.forward_key(view_k)

        # NTXentLoss returns the loss directly; it manages the memory bank
        loss = self.criterion(q, k)
        return loss