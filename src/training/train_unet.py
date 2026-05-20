
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import MultiStepLR
from pathlib import Path
from tqdm import tqdm
import numpy as np
import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.unet import UNet
from src.models.losses import topk_CE, iou_coefficient, \
    dice_coefficient, sensitivity_score
from src.dataset.seg_dataset import SegDataset, load_fold_paths

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
ALL_DIR     = Path("data/patches/segmentation/all")
TEST_DIR    = Path("data/patches/segmentation/test")
CKPT_DIR    = Path("checkpoints")
CKPT_DIR.mkdir(exist_ok=True)

BATCH_SIZE  = 8
EPOCHS      = 300
LR          = 0.001
MOMENTUM    = 0.99
MILESTONE   = 150
GAMMA       = 0.1
N_FOLDS     = 10
NUM_WORKERS = 4



def sanity_check(images, masks) -> bool:
    print("\n" + "="*55)
    print("  SANITY CHECK — overfit 10 positive patches")
    print("="*55)

    n       = min(10, len(images))
    dataset = SegDataset(images[:n], masks[:n], augment=False)
    loader  = DataLoader(dataset, batch_size=n)
    imgs, msks = next(iter(loader))
    imgs, msks = imgs.to(DEVICE), msks.to(DEVICE)

    model   = UNet().to(DEVICE)
    opt     = optim.SGD(model.parameters(), lr=0.01, momentum=0.99)
    loss_fn = topk_CE()

    model.train()
    for step in range(1, 101):
        pred = model(imgs)
        loss = loss_fn(pred, msks)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 20 == 0:
            with torch.no_grad():
                iou = iou_coefficient(pred, msks)
            print(f"  step {step:3d} | loss={loss.item():.4f} "
                  f"| IoU={iou:.4f}")

    with torch.no_grad():
        final_iou = iou_coefficient(model(imgs), msks)
    passed = final_iou > 0.5
    status = "✅ PASSED" if passed else "❌ FAILED"
    print(f"  {status} (IoU={final_iou:.4f})")
    print("="*55 + "\n")
    del model
    return passed


def save_debug(model, images, masks, path, n=4):
    model.eval()
    n  = min(n, len(images))
    ds = SegDataset(images[:n], masks[:n], augment=False)

    fig, axes = plt.subplots(n, 3, figsize=(9, 3*n))
    if n == 1:
        axes = [axes]

    for row in range(n):
        img, msk = ds[row]
        t = img.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            prob   = torch.sigmoid(model(t)).cpu().numpy()[0, 0]
        binary = (prob > 0.5).astype(np.float32)
        msk_np = msk.numpy()[0]
        dice   = (2*(binary*msk_np).sum()+1) / \
                 (binary.sum()+msk_np.sum()+1)

        axes[row][0].imshow(img.numpy()[0], cmap="gray")
        axes[row][0].set_title("Image"); axes[row][0].axis("off")
        axes[row][1].imshow(msk_np, cmap="gray")
        axes[row][1].set_title("GT"); axes[row][1].axis("off")
        axes[row][2].imshow(binary, cmap="gray")
        axes[row][2].set_title(f"Pred Dice={dice:.3f}")
        axes[row][2].axis("off")

    plt.tight_layout()
    plt.savefig(str(path), dpi=80, bbox_inches="tight")
    plt.close()
    print(f"  Debug → {path}")
    model.train()



def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total = 0.0
    bar   = tqdm(loader, desc="  Train", leave=False)
    for imgs, masks in bar:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        preds = model(imgs)
        loss  = criterion(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
        bar.set_postfix(loss=f"{loss.item():.4f}")
    return total / len(loader)



def validate(model, loader, criterion):
    model.eval()
    tot_loss = tot_iou = tot_dice = tot_sens = 0.0
    n = 0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            preds = model(imgs)
            tot_loss += criterion(preds, masks).item()
            tot_iou  += iou_coefficient(preds, masks)
            tot_dice += dice_coefficient(preds, masks)
            tot_sens += sensitivity_score(preds, masks)
            n += 1
    n = max(n, 1)
    return tot_loss/n, tot_iou/n, tot_dice/n, tot_sens/n



def evaluate_test(model_path: str):
    print(f"\n{'='*55}")
    print(f"  TEST SET EVALUATION")
    print(f"{'='*55}")

    test_imgs, test_masks = load_fold_paths(TEST_DIR)
    if not test_imgs:
        print("  No test patches found.")
        return

    test_ds = SegDataset(test_imgs, test_masks, augment=False)
    loader  = DataLoader(test_ds, batch_size=BATCH_SIZE,
                         num_workers=NUM_WORKERS)

    model = UNet().to(DEVICE)
    ckpt  = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    criterion = topk_CE()
    _, iou, dice, sens = validate(model, loader, criterion)

    print(f"  Test IoU        : {iou:.4f}")
    print(f"  Test Dice       : {dice:.4f}")
    print(f"  Test Sensitivity: {sens:.4f}")


    save_debug(model, test_imgs, test_masks,
               CKPT_DIR/"debug_test_final.png")

 
    results_path = Path("outputs/evaluation")
    results_path.mkdir(parents=True, exist_ok=True)
    with open(str(results_path/"seg_test_results.txt"), "w") as f:
        f.write(f"Test IoU        : {iou:.4f}\n")
        f.write(f"Test Dice       : {dice:.4f}\n")
        f.write(f"Test Sensitivity: {sens:.4f}\n")
    print(f"  Results saved → outputs/evaluation/seg_test_results.txt")


def train_fold(fold: int, train_imgs, train_masks,
               val_imgs, val_masks) -> tuple:
    print(f"\n{'='*55}")
    print(f"  FOLD {fold}/{N_FOLDS} | "
          f"train={len(train_imgs)} | val={len(val_imgs)}")
    print(f"{'='*55}")

    train_ds = SegDataset(train_imgs, train_masks, augment=True)
    val_ds   = SegDataset(val_imgs,   val_masks,   augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )

    model     = UNet(in_channels=1, out_channels=1, base_ch=64).to(DEVICE)
    criterion = topk_CE()
    optimizer = optim.SGD(model.parameters(),
                          lr=LR, momentum=MOMENTUM)
    scheduler = MultiStepLR(optimizer,
                            milestones=[MILESTONE], gamma=GAMMA)

    best_iou  = 0.0
    ckpt_path = CKPT_DIR / f"unet_fold{fold}.pth"

    for epoch in range(1, EPOCHS + 1):
        tr_loss              = train_epoch(model, train_loader,
                                           optimizer, criterion)
        val_loss, val_iou, \
        val_dice, val_sens   = validate(model, val_loader, criterion)
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        print(f"  [{fold}] Ep {epoch:3d}/{EPOCHS} | "
              f"tr={tr_loss:.4f} | "
              f"val={val_loss:.4f} | "
              f"IoU={val_iou:.4f} | "
              f"Dice={val_dice:.4f} | "
              f"Sens={val_sens:.4f} | "
              f"lr={lr:.2e}")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({
                "epoch"              : epoch,
                "fold"               : fold,
                "model_state_dict"   : model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_iou"            : best_iou,
                "val_dice"           : val_dice,
            }, str(ckpt_path))
            print(f"  ✅ Fold {fold} best saved "
                  f"(IoU={best_iou:.4f})")

        
        if epoch % 10 == 0:
            save_debug(
                model, val_imgs[:4], val_masks[:4],
                CKPT_DIR / f"debug_fold{fold}_ep{epoch:03d}.png"
            )

    print(f"  Fold {fold} done. Best IoU: {best_iou:.4f}")
    return best_iou, str(ckpt_path)


def main():
    print(f"Device : {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : "
              f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    
    all_imgs, all_masks = load_fold_paths(ALL_DIR)
    print(f"\nTotal CV patches: {len(all_imgs)}")

    
    if not sanity_check(all_imgs, all_masks):
        print("WARNING: sanity check failed — "
              "check data pipeline before proceeding")

    # 10-fold CV
    kf         = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_ious  = []
    best_ckpt  = None
    best_iou   = 0.0

    for fold, (tr_idx, val_idx) in \
            enumerate(kf.split(all_imgs), start=1):

        tr_imgs   = [all_imgs[i]  for i in tr_idx]
        tr_masks  = [all_masks[i] for i in tr_idx]
        val_imgs  = [all_imgs[i]  for i in val_idx]
        val_masks = [all_masks[i] for i in val_idx]

        fold_iou, ckpt_path = train_fold(
            fold, tr_imgs, tr_masks, val_imgs, val_masks
        )
        fold_ious.append(fold_iou)

        if fold_iou > best_iou:
            best_iou  = fold_iou
            best_ckpt = ckpt_path

    
    print(f"\n{'='*55}")
    print(f"  10-FOLD CV RESULTS")
    print(f"{'='*55}")
    for i, iou in enumerate(fold_ious, 1):
        marker = " ← best" if iou == max(fold_ious) else ""
        print(f"  Fold {i:2d}: IoU={iou:.4f}{marker}")
    print(f"\n  Mean IoU : {np.mean(fold_ious):.4f}")
    print(f"  Std  IoU : {np.std(fold_ious):.4f}")
    print(f"  Best fold: {np.argmax(fold_ious)+1} "
          f"(IoU={max(fold_ious):.4f})")

    
    best_path = str(CKPT_DIR / "unet_best.pth")
    shutil.copy(best_ckpt, best_path)
    print(f"\n  Best checkpoint → checkpoints/unet_best.pth")

    
    evaluate_test(best_path)


if __name__ == "__main__":
    main()