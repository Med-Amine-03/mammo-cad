"""
train_classifier.py — EfficientNet-B3 two-stage training  (v5)
================================================================
Changes vs v4:
  [FIX-7]  LR scheduler clip: plateau.step(min(val_loss, 2.0))
           Val_loss spikes (29, 8, 3) from hard batches triggered
           ReduceLROnPlateau prematurely. LR decayed to 3.9e-08 by
           epoch 96 — model stopped learning too early.
           Capping at 2.0 means spikes are invisible to the scheduler.
           Normal loss (0.65–1.5) still drives decay correctly.

  [FIX-8]  patience_stop Stage 2: 25 → 35
           With cleaner LR decay, model trains longer before plateauing.
           More patience lets it fully exploit the 150-epoch budget.

All v4 fixes kept:
  [FIX-1]  validate() in full float32 — no autocast
  [FIX-2]  NaN-safe label conversion in compute_metrics
  [FIX-3]  NaN-safe early stopping via update_early_stop()
  [FIX-4]  LR scheduler only steps on finite val_loss
  [FIX-5]  pos_weight cap 2.5
  [FIX-6]  Label smoothing 0.05
  MixUp alpha=0.2, SWA 20 epochs, AdamW, BN re-enable, warmup

Run:
  python src/training/train_classifier.py --stage 1
  python src/training/train_classifier.py --stage 2
  python src/training/train_classifier.py --stage 2 --eval_test
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import GradScaler, autocast
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score, roc_curve
from collections import Counter

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.efficientnet import EfficientNetClassifier
from src.dataset.cls_dataset import ClsDataset


# ─────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = Path("data/patches/classification")
CKPT_DIR = Path("checkpoints")
CKPT_DIR.mkdir(exist_ok=True)

POS_WEIGHT_CAP = 2.5

# [FIX-7] Loss ceiling for scheduler — spikes above this are ignored
SCHEDULER_LOSS_CLIP = 2.0

STAGE1 = dict(
    epochs        = 60,
    lr            = 1e-4,
    batch         = 32,
    grad_clip     = 1.0,
    patience_lr   = 7,
    patience_stop = 20,
    label_smooth  = 0.05,
    mixup_alpha   = 0.2,
    ckpt          = "efficientnet_stage1.pth",
)
STAGE2 = dict(
    epochs        = 150,
    lr_bb         = 1e-5,
    lr_head       = 1e-4,
    warmup_epochs = 5,
    batch         = 32,
    grad_clip     = 0.5,
    patience_lr   = 7,
    patience_stop = 35,    # [FIX-8] raised from 25
    label_smooth  = 0.05,
    mixup_alpha   = 0.2,
    swa_epochs    = 20,
    swa_lr        = 1e-6,
    ckpt          = "efficientnet_stage2.pth",
    ckpt_swa      = "efficientnet_stage2_swa.pth",
)


# ─────────────────────────────────────────────────────────────────────
# MIXUP
# ─────────────────────────────────────────────────────────────────────
def mixup_batch(imgs: torch.Tensor,
                labels: torch.Tensor,
                alpha: float = 0.2) -> tuple[torch.Tensor, torch.Tensor]:
    if alpha <= 0:
        return imgs, labels
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(imgs.size(0), device=imgs.device)
    return (lam * imgs + (1.0 - lam) * imgs[idx],
            lam * labels + (1.0 - lam) * labels[idx])


# ─────────────────────────────────────────────────────────────────────
# LABEL SMOOTHING BCE
# ─────────────────────────────────────────────────────────────────────
class LabelSmoothingBCE(nn.Module):
    def __init__(self, pos_weight: torch.Tensor | None = None,
                 smoothing: float = 0.05):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight, reduction="none"
        )

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        targets_s = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        loss = self.bce(logits, targets_s).mean()
        if not torch.isfinite(loss):
            return torch.tensor(0.7, device=logits.device,
                                dtype=logits.dtype, requires_grad=True)
        return loss


# ─────────────────────────────────────────────────────────────────────
# NaN-SAFE EARLY STOPPING
# ─────────────────────────────────────────────────────────────────────
def update_early_stop(val_loss: float,
                      best_val_loss: float,
                      no_improve: int) -> tuple[float, int]:
    if not np.isfinite(val_loss):
        return best_val_loss, no_improve
    if val_loss < best_val_loss:
        return val_loss, 0
    return best_val_loss, no_improve + 1


# ─────────────────────────────────────────────────────────────────────
# PATIENT SAMPLER
# ─────────────────────────────────────────────────────────────────────
def make_patient_sampler(dataset: ClsDataset) -> WeightedRandomSampler:
    patient_ids = []
    for path, _ in dataset.samples:
        stem  = path.stem
        parts = stem.rsplit("_", 1)
        patient_ids.append(parts[0] if len(parts) == 2 else stem)

    counts  = Counter(patient_ids)
    n_pats  = len(counts)
    weights = torch.tensor(
        [1.0 / counts[p] for p in patient_ids], dtype=torch.float
    )
    print(f"  Patient sampler: {n_pats} unique patients / "
          f"{len(dataset)} total crops")
    return WeightedRandomSampler(
        weights, num_samples=n_pats, replacement=False
    )


# ─────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────
def compute_metrics(all_labels: list, all_probs: list,
                    threshold: float = 0.5) -> dict:
    probs_arr = np.array(all_probs, dtype=np.float64)
    probs_arr = np.where(np.isfinite(probs_arr), probs_arr,
                         0.5 * np.ones_like(probs_arr))
    probs_arr = np.clip(probs_arr, 0.0, 1.0)
    preds     = (probs_arr > threshold).astype(int)
    labels    = np.array(
        [int(round(float(l))) if np.isfinite(float(l)) else 0
         for l in all_labels], dtype=int
    )
    tn, fp, fn, tp = confusion_matrix(
        labels, preds, labels=[0, 1]
    ).ravel()
    auc_val = (roc_auc_score(labels, probs_arr)
               if len(set(labels.tolist())) > 1 else 0.5)
    return dict(
        auc  = auc_val,
        acc  = (tp + tn) / (tp + tn + fp + fn + 1e-8),
        sens = tp / (tp + fn + 1e-8),
        spec = tn / (tn + fp + 1e-8),
        f1   = f1_score(labels, preds, zero_division=0),
        tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn),
    )


def find_best_threshold(all_labels: list,
                        all_probs:  list) -> tuple[float, float]:
    best_f1, best_thr = 0.0, 0.5
    for thr in np.linspace(0.1, 0.9, 81):
        f1 = f1_score(
            (np.array(all_probs) > thr).astype(int),
            np.array([int(round(float(l))) for l in all_labels], dtype=int),
            zero_division=0,
        )
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr, best_f1


# ─────────────────────────────────────────────────────────────────────
# TRAIN EPOCH
# ─────────────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion,
                scaler, grad_clip: float,
                mixup_alpha: float = 0.2) -> tuple[float, float]:
    model.train()
    total_loss = correct = total = 0

    for imgs, labels in tqdm(loader, desc="  train", leave=False):
        imgs   = imgs.to(DEVICE,   non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        imgs, labels = mixup_batch(imgs, labels, alpha=mixup_alpha)

        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda"):
            logits = model(imgs)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        with torch.no_grad():
            hard = (labels > 0.5).float()
            correct += ((torch.sigmoid(logits) > 0.5).float()
                        == hard).sum().item()
            total   += labels.size(0)

    return total_loss / len(loader), correct / max(total, 1)


# ─────────────────────────────────────────────────────────────────────
# VALIDATE — full float32, no autocast
# ─────────────────────────────────────────────────────────────────────
def validate(model, loader, criterion) -> tuple:
    model.eval()
    total_loss = 0.0
    all_probs:  list[float] = []
    all_labels: list[int]   = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            # Full float32 — eliminates every NaN source
            logits    = model(imgs.float())
            loss      = criterion(logits, labels)
            loss_item = loss.item()
            total_loss += loss_item if np.isfinite(loss_item) else 0.7

            probs = torch.sigmoid(logits)
            probs = torch.clamp(probs, 0.0, 1.0).cpu()

            label_list = labels.cpu().numpy().tolist()
            label_list = [
                int(round(float(v))) if np.isfinite(float(v)) else 0
                for v in label_list
            ]
            all_probs.extend(probs.numpy().flatten().tolist())
            all_labels.extend(label_list)

    avg_loss = total_loss / len(loader)
    m = compute_metrics(all_labels, all_probs)
    return avg_loss, m, all_probs, all_labels


# ─────────────────────────────────────────────────────────────────────
# STAGE 1
# ─────────────────────────────────────────────────────────────────────
def stage1() -> None:
    print("\n" + "=" * 60)
    print("  STAGE 1 — EfficientNet-B3 | head + BN only")
    print("=" * 60)
    cfg = STAGE1

    train_ds = ClsDataset(DATA_DIR, "train", augment=True)
    val_ds   = ClsDataset(DATA_DIR, "val",   augment=False)
    sampler  = make_patient_sampler(train_ds)

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch"], sampler=sampler,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch"] * 2, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )

    model = EfficientNetClassifier(freeze=True).to(DEVICE)
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            for p in m.parameters():
                p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)
    total_p   = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total_p:,} "
          f"({100 * trainable / total_p:.1f}%)")

    raw_pw = train_ds.pos_weight
    pw_val = min(raw_pw, POS_WEIGHT_CAP)
    print(f"  pos_weight: {raw_pw:.4f} → capped at {pw_val:.4f}")
    pos_weight = torch.tensor([pw_val], device=DEVICE)

    criterion = LabelSmoothingBCE(
        pos_weight=pos_weight, smoothing=cfg["label_smooth"]
    )
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["lr"], weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5,
        patience=cfg["patience_lr"], min_lr=1e-7,
    )
    scaler = GradScaler("cuda")

    best_auc      = 0.0
    best_val_loss = float("inf")
    no_improve    = 0
    ckpt_path     = CKPT_DIR / cfg["ckpt"]
    history       = {"tr_loss": [], "val_loss": [], "val_auc": [], "lr": []}

    for epoch in range(1, cfg["epochs"] + 1):
        tr_loss, tr_acc     = train_epoch(
            model, train_loader, optimizer, criterion,
            scaler, cfg["grad_clip"], cfg["mixup_alpha"]
        )
        val_loss, m, vp, vl = validate(model, val_loader, criterion)

        # [FIX-7] Clip before scheduler — spikes don't trigger decay
        if np.isfinite(val_loss):
            scheduler.step(min(val_loss, SCHEDULER_LOSS_CLIP))
        lr = optimizer.param_groups[0]["lr"]

        history["tr_loss"].append(tr_loss)
        history["val_loss"].append(
            val_loss if np.isfinite(val_loss) else float("nan")
        )
        history["val_auc"].append(m["auc"])
        history["lr"].append(lr)

        auc_flag = ""
        if m["auc"] > best_auc:
            best_auc = m["auc"]
            torch.save({
                "epoch": epoch, "stage": 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": best_auc, "metrics": m,
                "val_probs": vp, "val_labels": vl,
                "pos_weight": pw_val,
            }, str(ckpt_path))
            auc_flag = " ✅"

        best_val_loss, no_improve = update_early_stop(
            val_loss, best_val_loss, no_improve
        )

        loss_str = "nan[skip]" if not np.isfinite(val_loss) \
                   else f"{val_loss:.4f}"
        print(f"  Ep {epoch:3d}/{cfg['epochs']} | "
              f"tr={tr_loss:.4f}/{tr_acc:.3f} | "
              f"val_loss={loss_str} AUC={m['auc']:.4f} "
              f"Sens={m['sens']:.3f} Spec={m['spec']:.3f} "
              f"F1={m['f1']:.3f} lr={lr:.1e}{auc_flag}")

        if no_improve >= cfg["patience_stop"]:
            print(f"\n  Early stop at epoch {epoch}")
            break

    _save_curves(history, CKPT_DIR / "stage1_curves.png", "Stage 1")
    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location="cpu",
                          weights_only=False)
        thr, f1 = find_best_threshold(
            ckpt["val_labels"], ckpt["val_probs"]
        )
        print(f"\n  Best val AUC={best_auc:.4f} | "
              f"threshold={thr:.2f} (F1={f1:.4f})")
    print(f"\n  Next: --stage 2")


# ─────────────────────────────────────────────────────────────────────
# STAGE 2
# ─────────────────────────────────────────────────────────────────────
def stage2(eval_test: bool = False) -> None:
    print("\n" + "=" * 60)
    print("  STAGE 2 — EfficientNet-B3 | blocks 5-8 + SWA")
    print("=" * 60)
    cfg = STAGE2

    s1_path = CKPT_DIR / STAGE1["ckpt"]
    if not s1_path.exists():
        raise FileNotFoundError(
            f"Stage 1 checkpoint not found: {s1_path}\nRun --stage 1 first."
        )

    model = EfficientNetClassifier(freeze=True).to(DEVICE)
    s1    = torch.load(str(s1_path), map_location=DEVICE,
                       weights_only=False)
    model.load_state_dict(s1["model_state_dict"])
    print(f"\n  Loaded Stage 1 (val AUC={s1['val_auc']:.4f})")

    model.unfreeze_stage2()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            for p in m.parameters():
                p.requires_grad = True

    train_ds = ClsDataset(DATA_DIR, "train", augment=True)
    val_ds   = ClsDataset(DATA_DIR, "val",   augment=False)
    sampler  = make_patient_sampler(train_ds)

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch"], sampler=sampler,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch"] * 2, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )

    raw_pw = train_ds.pos_weight
    pw_val = min(raw_pw, POS_WEIGHT_CAP)
    print(f"  pos_weight: {raw_pw:.4f} → capped at {pw_val:.4f}")
    pos_weight = torch.tensor([pw_val], device=DEVICE)

    criterion = LabelSmoothingBCE(
        pos_weight=pos_weight, smoothing=cfg["label_smooth"]
    )

    bb_params   = [p for n, p in model.net.named_parameters()
                   if p.requires_grad and "classifier" not in n]
    head_params = list(model.net.classifier.parameters())

    optimizer = optim.AdamW([
        {"params": bb_params,   "lr": 1e-8,          "weight_decay": 1e-4},
        {"params": head_params, "lr": cfg["lr_head"], "weight_decay": 1e-3},
    ])
    plateau = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5,
        patience=cfg["patience_lr"], min_lr=1e-8,
    )
    scaler = GradScaler("cuda")

    best_auc      = s1["val_auc"]
    best_val_loss = float("inf")
    no_improve    = 0
    ckpt_path     = CKPT_DIR / cfg["ckpt"]
    history       = {
        "tr_loss": [], "val_loss": [], "val_auc": [],
        "lr_bb": [], "lr_head": [],
    }

    for epoch in range(1, cfg["epochs"] + 1):
        if epoch <= cfg["warmup_epochs"]:
            frac = epoch / cfg["warmup_epochs"]
            optimizer.param_groups[0]["lr"] = cfg["lr_bb"] * frac
            warmup_tag = f" [warmup {epoch}/{cfg['warmup_epochs']}]"
        else:
            warmup_tag = ""

        tr_loss, tr_acc     = train_epoch(
            model, train_loader, optimizer, criterion,
            scaler, cfg["grad_clip"], cfg["mixup_alpha"]
        )
        val_loss, m, vp, vl = validate(model, val_loader, criterion)

        # [FIX-7] Clip before scheduler
        if epoch > cfg["warmup_epochs"] and np.isfinite(val_loss):
            plateau.step(min(val_loss, SCHEDULER_LOSS_CLIP))

        lrs = [pg["lr"] for pg in optimizer.param_groups]
        history["tr_loss"].append(tr_loss)
        history["val_loss"].append(
            val_loss if np.isfinite(val_loss) else float("nan")
        )
        history["val_auc"].append(m["auc"])
        history["lr_bb"].append(lrs[0])
        history["lr_head"].append(lrs[1])

        auc_flag = ""
        if m["auc"] > best_auc:
            best_auc = m["auc"]
            torch.save({
                "epoch": epoch, "stage": 2,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": best_auc, "metrics": m,
                "val_probs": vp, "val_labels": vl,
                "pos_weight": pw_val,
            }, str(ckpt_path))
            auc_flag = " ✅"

        best_val_loss, no_improve = update_early_stop(
            val_loss, best_val_loss, no_improve
        )

        loss_str = "nan[skip]" if not np.isfinite(val_loss) \
                   else f"{val_loss:.4f}"
        print(f"  Ep {epoch:3d}/{cfg['epochs']}{warmup_tag} | "
              f"tr={tr_loss:.4f}/{tr_acc:.3f} | "
              f"val_loss={loss_str} AUC={m['auc']:.4f} "
              f"Sens={m['sens']:.3f} Spec={m['spec']:.3f} "
              f"F1={m['f1']:.3f} bb_lr={lrs[0]:.1e}{auc_flag}")

        if (no_improve >= cfg["patience_stop"]
                and epoch > cfg["warmup_epochs"]):
            print(f"\n  Early stop at epoch {epoch}")
            break

    _save_curves(
        history, CKPT_DIR / "stage2_curves.png", "Stage 2",
        lr_keys=["lr_bb", "lr_head"],
    )

    # ── SWA ──────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  SWA — {cfg['swa_epochs']} epochs at lr={cfg['swa_lr']:.0e}")
    print(f"{'─'*60}")

    if ckpt_path.exists():
        best_ckpt = torch.load(str(ckpt_path), map_location=DEVICE,
                               weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])
        print(f"  Starting from best S2 "
              f"(val AUC={best_ckpt['val_auc']:.4f})")

    swa_model     = AveragedModel(model)
    swa_optimizer = optim.SGD(model.parameters(),
                              lr=cfg["swa_lr"], momentum=0.9)
    swa_scheduler = SWALR(swa_optimizer, swa_lr=cfg["swa_lr"],
                          anneal_epochs=5)

    swa_loader = DataLoader(
        train_ds, batch_size=cfg["batch"], shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )

    for swa_ep in range(1, cfg["swa_epochs"] + 1):
        model.train()
        ep_loss = 0.0
        for imgs, labels in tqdm(
            swa_loader,
            desc=f"  SWA ep {swa_ep}/{cfg['swa_epochs']}",
            leave=False,
        ):
            imgs   = imgs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            swa_optimizer.zero_grad(set_to_none=True)
            with autocast("cuda"):
                logits = model(imgs)
                loss   = criterion(logits, labels)
            loss.backward()
            swa_optimizer.step()
            ep_loss += loss.item()

        swa_model.update_parameters(model)
        swa_scheduler.step()
        print(f"  SWA ep {swa_ep}/{cfg['swa_epochs']} | "
              f"loss={ep_loss/len(swa_loader):.4f}")

    print("  Updating BatchNorm statistics...")
    update_bn(swa_loader, swa_model, device=DEVICE)

    _, m_swa, vp_swa, vl_swa = validate(
        swa_model, val_loader, criterion
    )
    print(f"\n  SWA val AUC = {m_swa['auc']:.4f} "
          f"(best S2 = {best_auc:.4f})")

    swa_ckpt_path = CKPT_DIR / cfg["ckpt_swa"]
    thr_swa, _    = find_best_threshold(vl_swa, vp_swa)
    torch.save({
        "stage": "2_swa",
        "model_state_dict": swa_model.module.state_dict(),
        "val_auc": m_swa["auc"], "metrics": m_swa,
        "val_probs": vp_swa, "val_labels": vl_swa,
        "pos_weight": pw_val, "threshold": thr_swa,
    }, str(swa_ckpt_path))
    print(f"  SWA saved: {swa_ckpt_path}")

    if m_swa["auc"] >= best_auc:
        torch.save({
            "epoch": "swa", "stage": 2,
            "model_state_dict": swa_model.module.state_dict(),
            "val_auc": m_swa["auc"], "metrics": m_swa,
            "val_probs": vp_swa, "val_labels": vl_swa,
            "pos_weight": pw_val,
        }, str(ckpt_path))
        print(f"  SWA overwrote S2 best "
              f"(AUC {m_swa['auc']:.4f} >= {best_auc:.4f})")
        best_auc = m_swa["auc"]

    print(f"\n{'='*60}")
    print(f"  Done | Best val AUC = {best_auc:.4f}")
    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location="cpu",
                          weights_only=False)
        thr, f1 = find_best_threshold(
            ckpt["val_labels"], ckpt["val_probs"]
        )
        print(f"  Threshold = {thr:.2f} (F1={f1:.4f})")
    print(f"{'='*60}")
    print(f"\n  Next: python src/inference/evaluate_cls.py")

    if eval_test:
        _evaluate_test(model, ckpt_path)


# ─────────────────────────────────────────────────────────────────────
# TEST EVALUATION
# ─────────────────────────────────────────────────────────────────────
def _evaluate_test(model, ckpt_path: Path) -> None:
    print("\n" + "─" * 60)
    print("  FINAL TEST EVALUATION")
    print("─" * 60)

    ckpt = torch.load(str(ckpt_path), map_location=DEVICE,
                      weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    best_thr, _ = find_best_threshold(
        ckpt["val_labels"], ckpt["val_probs"]
    )

    test_ds    = ClsDataset(DATA_DIR, "test", augment=False)
    raw_pw     = test_ds.pos_weight
    pw_val     = min(raw_pw, POS_WEIGHT_CAP)
    pos_weight = torch.tensor([pw_val], device=DEVICE)
    criterion  = LabelSmoothingBCE(pos_weight=pos_weight, smoothing=0.0)
    loader     = DataLoader(
        test_ds, batch_size=64, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    _, m05, probs, labels = validate(model, loader, criterion)
    m_thr = compute_metrics(labels, probs, threshold=best_thr)

    for tag, m in [("0.50 (default)", m05),
                   (f"{best_thr:.2f} (val-optimised)", m_thr)]:
        print(f"\n  Threshold {tag}")
        print(f"  AUC={m['auc']:.4f}  Acc={m['acc']:.4f}  "
              f"Sens={m['sens']:.4f}  Spec={m['spec']:.4f}  "
              f"F1={m['f1']:.4f}")
        print(f"  TP={m['tp']}  TN={m['tn']}  "
              f"FP={m['fp']}  FN={m['fn']}")

    auc = m05["auc"]
    print(f"\n  Published range (calc+mass): 0.78–0.84")
    if   auc >= 0.84: print("  Status: above published range")
    elif auc >= 0.80: print("  Status: top of range — target met")
    elif auc >= 0.78: print("  Status: within published range")
    else:             print("  Status: below target")

    fpr, tpr, _ = roc_curve(labels, probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, "b-", lw=2,
            label=f"EfficientNet-B3 (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.axhline(0.8, color="orange", ls="--", lw=0.8, label="0.80 target")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC — Test Set"); ax.legend()
    plt.tight_layout()
    plt.savefig(str(CKPT_DIR / "test_roc_curve.png"), dpi=120)
    plt.close()

    with open(CKPT_DIR / "test_results.txt", "w") as f:
        f.write("CLASSIFICATION RESULTS — mammo-cad v5\n")
        f.write("Model    : EfficientNet-B3 (dropout=0.5) + MixUp + SWA\n")
        f.write("Data     : CBIS-DDSM DICOM (calc+mass, clean split)\n")
        f.write(f"Threshold: {best_thr:.2f}\n\n")
        for k, v in m_thr.items():
            f.write(f"{k}: {v}\n")

    print(f"\n  Saved: checkpoints/test_roc_curve.png")
    print(f"  Saved: checkpoints/test_results.txt")


# ─────────────────────────────────────────────────────────────────────
# CURVES
# ─────────────────────────────────────────────────────────────────────
def _save_curves(history: dict, path: Path, title: str,
                 lr_keys: list | None = None) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(title)

    axes[0].plot(history["tr_loss"], label="train")
    finite_pairs = [(i, v) for i, v in enumerate(history["val_loss"])
                    if v is not None and np.isfinite(v)]
    if finite_pairs:
        idx, vals = zip(*finite_pairs)
        axes[0].plot(idx, vals, label="val (finite only)")
        p95 = np.percentile(vals, 95)
        axes[0].set_ylim(0, min(5, p95 * 1.5))
    axes[0].set_title("Loss"); axes[0].legend()

    axes[1].plot(history["val_auc"], color="steelblue", label="val AUC")
    axes[1].axhline(0.5,  color="red",    ls="--", lw=0.8, label="random")
    axes[1].axhline(0.80, color="orange", ls="--", lw=0.8, label="0.80 target")
    axes[1].axhline(0.84, color="green",  ls="--", lw=0.8, label="0.84 stretch")
    axes[1].set_ylim(0.4, 1.0)
    axes[1].set_title("Val AUC-ROC"); axes[1].legend(fontsize=8)

    for k in (lr_keys or ["lr"]):
        if k in history:
            axes[2].plot(history[k], label=k)
    axes[2].set_title("Learning Rate")
    axes[2].set_yscale("log"); axes[2].legend()

    plt.tight_layout()
    plt.savefig(str(path), dpi=100)
    plt.close()
    print(f"  Curves saved: {path}")


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=[1, 2], required=True)
    parser.add_argument("--eval_test", action="store_true")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
    print(f"MixUp : alpha={STAGE1['mixup_alpha']}")
    print(f"SWA   : {STAGE2['swa_epochs']} epochs (Stage 2 only)")
    print(f"PW cap: {POS_WEIGHT_CAP}")
    print(f"LR clip: val_loss capped at {SCHEDULER_LOSS_CLIP} for scheduler")

    if args.stage == 1:
        stage1()
    else:
        stage2(eval_test=args.eval_test)