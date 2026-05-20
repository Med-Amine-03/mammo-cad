
import torch
import torch.nn as nn


class topk_CE(nn.Module):
   
    def __init__(self, neg_ratio: int = 3):
        super().__init__()
        self.neg_ratio = neg_ratio
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        targets   = targets.float()
        loss_map  = self.bce(logits, targets)   # (B, 1, H, W)
        B         = logits.size(0)
        collected = []

        for i in range(B):
            l = loss_map[i, 0]   # (H, W)
            t = targets[i, 0]    # (H, W)

            pos_loss = l[t == 1]
            neg_loss = l[t == 0]
            n_pos    = len(pos_loss)

            if n_pos > 0:
               
                collected.append(pos_loss)
                
                n_keep = min(self.neg_ratio * n_pos, len(neg_loss))
                if n_keep > 0:
                    hard_neg, _ = torch.topk(neg_loss, n_keep)
                    collected.append(hard_neg)
            else:
                
                n_keep = min(100, len(neg_loss))
                if n_keep > 0:
                    hard_neg, _ = torch.topk(neg_loss, n_keep)
                    collected.append(hard_neg)

        if not collected:
            return loss_map.mean()

        return torch.cat(collected).mean()


# ─────────────────────────────────────────────────────────
# Evaluation metrics 
# ─────────────────────────────────────────────────────────

def dice_coefficient(logits: torch.Tensor,
                     targets: torch.Tensor,
                     threshold: float = 0.5,
                     smooth: float = 1.0) -> float:
    probs  = torch.sigmoid(logits)
    preds  = (probs > threshold).float()
    flat_p = preds.view(preds.size(0), -1)
    flat_t = targets.float().view(targets.size(0), -1)
    inter  = (flat_p * flat_t).sum(dim=1)
    union  = flat_p.sum(dim=1) + flat_t.sum(dim=1)
    dice   = (2.0 * inter + smooth) / (union + smooth)
    return dice.mean().item()


def iou_coefficient(logits: torch.Tensor,
                    targets: torch.Tensor,
                    threshold: float = 0.5,
                    smooth: float = 1.0) -> float:
    probs  = torch.sigmoid(logits)
    preds  = (probs > threshold).float()
    flat_p = preds.view(preds.size(0), -1)
    flat_t = targets.float().view(targets.size(0), -1)
    inter  = (flat_p * flat_t).sum(dim=1)
    union  = flat_p.sum(dim=1) + flat_t.sum(dim=1) - inter
    iou    = (inter + smooth) / (union + smooth)
    return iou.mean().item()


def sensitivity_score(logits: torch.Tensor,
                      targets: torch.Tensor,
                      threshold: float = 0.5,
                      smooth: float = 1.0) -> float:
    
    probs  = torch.sigmoid(logits)
    preds  = (probs > threshold).float()
    flat_p = preds.view(preds.size(0), -1)
    flat_t = targets.float().view(targets.size(0), -1)
    tp     = (flat_p * flat_t).sum(dim=1)
    fn     = ((1 - flat_p) * flat_t).sum(dim=1)
    sens   = (tp + smooth) / (tp + fn + smooth)
    return sens.mean().item()