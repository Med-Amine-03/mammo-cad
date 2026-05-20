"""
efficientnet.py — EfficientNet-B3 binary classifier  (v5)
==========================================================
Change vs v4:
  [ART-1] Dropout 0.4 → 0.5  (Shia et al. Nature Sci Rep 2025)
           Independent paper on same task used p=0.5.
           With val→test gap still present, extra regularisation helps.

Architecture (unchanged):
  Backbone : EfficientNet-B3, ImageNet pretrained
  Head     : Linear(1536→256) → ReLU → Dropout(0.5) → Linear(256→1)
  Output   : raw logit — sigmoid applied externally

Stage 1: backbone frozen, head + BN train only
Stage 2: blocks 5,6,7,8 + head unfrozen, linear warmup 5 epochs
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights


class EfficientNetClassifier(nn.Module):

    def __init__(self, freeze: bool = True) -> None:
        super().__init__()
        self.net = efficientnet_b3(
            weights=EfficientNet_B3_Weights.IMAGENET1K_V1
        )

        # Replace classifier head
        in_features = self.net.classifier[1].in_features  # 1536
        self.net.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),   # [ART-1] raised from 0.4 → Shia et al. 2025
            nn.Linear(256, 1),
        )

        # Careful initialisation — avoids large logits at epoch 0
        nn.init.kaiming_normal_(
            self.net.classifier[0].weight, mode="fan_out"
        )
        nn.init.zeros_(self.net.classifier[0].bias)
        nn.init.xavier_normal_(self.net.classifier[3].weight)
        nn.init.zeros_(self.net.classifier[3].bias)

        if freeze:
            self._freeze_backbone()

    def _freeze_backbone(self) -> None:
        for p in self.net.parameters():
            p.requires_grad = False
        for p in self.net.classifier.parameters():
            p.requires_grad = True

    def unfreeze_stage2(self) -> None:
        """Unfreeze blocks 5,6,7,8 + classifier for Stage 2."""
        for p in self.net.features.parameters():
            p.requires_grad = False
        for idx in [5, 6, 7, 8]:
            for p in self.net.features[idx].parameters():
                p.requires_grad = True
        for p in self.net.classifier.parameters():
            p.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters()
                        if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  Unfrozen: blocks 5,6,7,8 + classifier")
        print(f"  Trainable: {trainable:,} / {total:,} "
              f"({100 * trainable / total:.1f}%)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)