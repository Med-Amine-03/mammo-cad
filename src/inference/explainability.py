"""
explainability.py — Grad-CAM++ + Deep SHAP for mammo-cad  (v1)
===============================================================

Two complementary XAI methods for EfficientNet-B3 classification:

┌─────────────────┬────────────────────────────────────────────────┐
│ Method          │ What it shows                                  │
├─────────────────┼────────────────────────────────────────────────┤
│ Grad-CAM++      │ WHERE the model looks (spatial activation map) │
│                 │ Upgrade over Grad-CAM: better localisation of  │
│                 │ small structures (microcalcifications).         │
│                 │ Target: net.features[7] (last MBConv, 7×7)     │
├─────────────────┼────────────────────────────────────────────────┤
│ Deep SHAP       │ WHY each pixel contributes to the decision     │
│                 │ Signed attribution: RED = pushes toward MAL,   │
│                 │ BLUE = pushes toward BEN.                      │
│                 │ Uses 50 background samples from the image.     │
└─────────────────┴────────────────────────────────────────────────┘

Usage
-----
  from explainability import ExplainabilityEngine

  engine = ExplainabilityEngine(clf.model, device)

  # Single crop
  result = engine.explain(crop_uint8_224x224)
  engine.plot_single(result, title="R01 — MALIGNANT P=0.72")

  # Full figure for all regions in a pipeline result
  engine.plot_regions(regions, crops, output_path="outputs/xai_panel.png")
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ImageNet normalisation (same as classify.py)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────
def _to_tensor(crop: np.ndarray,
               device: torch.device) -> torch.Tensor:
    """uint8 grayscale 224×224 → normalised [1,3,224,224] float32 tensor."""
    img_f  = crop.astype(np.float32) / 255.0
    img_3c = np.stack([img_f] * 3, axis=0)
    img_3c = (img_3c - _MEAN[:, None, None]) / _STD[:, None, None]
    return torch.from_numpy(img_3c).unsqueeze(0).float().to(device)


def _normalise_cam(cam: np.ndarray) -> np.ndarray:
    """Normalise a CAM map to [0, 1]."""
    mn, mx = cam.min(), cam.max()
    if mx > mn:
        return (cam - mn) / (mx - mn)
    return np.zeros_like(cam)


# ─────────────────────────────────────────────────────────────────────
# GRAD-CAM++
# ─────────────────────────────────────────────────────────────────────
class GradCAMPlusPlus:
    """
    Grad-CAM++ (Chattopadhay et al. 2018).

    Improvement over Grad-CAM:
      - Uses second and third-order gradients to weight each pixel
        of the feature map individually (not just channel-mean).
      - Better localisation of small, multiple, or overlapping objects.
      - Particularly suited for microcalcification detection where
        multiple tiny structures coexist in one patch.

    Target layer: model.net.features[7]
      → Last MBConv block before GlobalAvgPool.
      → Feature map shape: (B, 384, 7, 7).
      → Upsampled to 224×224 via bicubic interpolation.
    """

    def __init__(self, model: nn.Module,
                 target_layer: nn.Module) -> None:
        self.model  = model
        self._feats: Optional[torch.Tensor] = None
        self._grads: Optional[torch.Tensor] = None

        self._fwd = target_layer.register_forward_hook(self._save_feats)
        self._bwd = target_layer.register_full_backward_hook(self._save_grads)

    def _save_feats(self, m, i, o):
        self._feats = o

    def _save_grads(self, m, gi, go):
        self._grads = go[0]

    def remove(self):
        self._fwd.remove()
        self._bwd.remove()

    def generate(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Compute Grad-CAM++ map.

        Parameters
        ----------
        tensor : torch.Tensor  shape (1, 3, 224, 224)

        Returns
        -------
        cam : np.ndarray  float32 in [0,1], shape (224, 224)
        """
        self.model.eval()
        self.model.zero_grad()

        # Patch inplace activations for safe backward
        patched = _disable_inplace(self.model)

        try:
            tensor = tensor.clone().requires_grad_(True)
            logit  = self.model(tensor)
            score  = torch.sigmoid(logit)
            score.backward(retain_graph=False)
        finally:
            _restore_inplace(patched)

        # Grad-CAM++ weights
        # α_kc = (∂²y / ∂A_ij^k²) / (2·∂²y/∂A_ij^k² + Σ_ij A_ij·∂³y/∂A^k³)
        grads  = self._grads                    # (1, C, H, W)
        feats  = self._feats.detach()           # (1, C, H, W)

        grads_sq  = grads ** 2
        grads_cu  = grads ** 3
        sum_feats = feats.sum(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        denom   = 2.0 * grads_sq + sum_feats * grads_cu
        denom   = torch.where(denom != 0,
                              denom,
                              torch.ones_like(denom))
        alpha   = grads_sq / denom              # (1, C, H, W)

        # ReLU on gradients before weighting
        relu_grads = F.relu(grads)
        weights    = (alpha * relu_grads).sum(dim=(2, 3),
                                               keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of feature maps
        cam = (weights * feats).sum(dim=1).squeeze(0)  # (H, W)
        cam = F.relu(cam).cpu().numpy()
        cam = cv2.resize(cam, (224, 224),
                         interpolation=cv2.INTER_CUBIC)
        return _normalise_cam(cam)


# ─────────────────────────────────────────────────────────────────────
# DEEP SHAP
# ─────────────────────────────────────────────────────────────────────
def _disable_inplace(model: nn.Module) -> list:
    """
    Set inplace=False on all SiLU/ReLU activations.
    Returns list of (module, original_inplace) for restoration.
    """
    patched = []
    for m in model.modules():
        if isinstance(m, (nn.SiLU, nn.ReLU, nn.ReLU6, nn.Hardswish)):
            if getattr(m, 'inplace', False):
                patched.append((m, True))
                m.inplace = False
    return patched


def _restore_inplace(patched: list) -> None:
    """Restore inplace=True on previously patched activations."""
    for m, val in patched:
        m.inplace = val


class DeepSHAPExplainer:
    """
    Integrated Gradients attribution for EfficientNet-B3.

    Replaces Deep SHAP / GradientSHAP which are incompatible with
    EfficientNet-B3 due to SiLU inplace + output shape issues.

    Integrated Gradients (Sundararajan et al. 2017):
      - Computes the path integral of gradients from a baseline to the input.
      - Axiomatically correct: satisfies completeness, sensitivity, linearity.
      - Signed: positive = pushes toward MALIGNANT, negative = toward BENIGN.
      - Uses captum if available, falls back to manual implementation.

    This is the XAI method used in most medical imaging papers (equivalent
    to SHAP for gradient-based models, but more stable).
    """

    def __init__(self, model: nn.Module,
                 device: torch.device,
                 n_steps: int = 50) -> None:
        self.model   = model
        self.device  = device
        self.n_steps = n_steps

        # Try captum first, fall back to manual IG
        try:
            from captum.attr import IntegratedGradients
            self._captum_ig = IntegratedGradients
            self._use_captum = True
        except ImportError:
            self._use_captum = False

    def explain(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Compute Integrated Gradients attributions.

        Parameters
        ----------
        tensor : torch.Tensor  shape (1, 3, 224, 224)

        Returns
        -------
        attr : np.ndarray float32 (224, 224) — signed attributions.
        """
        self.model.eval()
        patched = _disable_inplace(self.model)

        try:
            if self._use_captum:
                attr = self._captum_explain(tensor)
            else:
                attr = self._manual_ig(tensor)
        finally:
            _restore_inplace(patched)

        return attr

    def _captum_explain(self, tensor: torch.Tensor) -> np.ndarray:
        """Integrated Gradients via captum."""
        # Wrapper: model output → scalar sigmoid score
        class _Wrap(nn.Module):
            def __init__(self, m): super().__init__(); self.m = m
            def forward(self, x):
                return torch.sigmoid(self.m(x))

        wrapped  = _Wrap(self.model).to(self.device)
        baseline = torch.zeros_like(tensor)

        ig   = self._captum_ig(wrapped)
        attr = ig.attribute(
            tensor,
            baselines     = baseline,
            n_steps       = self.n_steps,
            return_convergence_delta = False,
        )
        # attr shape: (1, 3, 224, 224) → average channels → (224, 224)
        attr = attr.squeeze(0).mean(dim=0).detach().cpu().numpy()

        # Smooth to reduce high-frequency gradient noise
        from scipy.ndimage import gaussian_filter
        attr = gaussian_filter(attr.astype(np.float32), sigma=3.0)
        return attr.astype(np.float32)

    def _manual_ig(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Manual Integrated Gradients — no captum dependency.
        IG = (input - baseline) × ∫₀¹ ∇f(baseline + α(input-baseline)) dα
        Approximated via Riemann sum with n_steps steps.
        """
        baseline   = torch.zeros_like(tensor)
        delta      = tensor - baseline
        integrated = torch.zeros_like(tensor)

        for k in range(1, self.n_steps + 1):
            alpha     = k / self.n_steps
            interp    = (baseline + alpha * delta).requires_grad_(True)
            logit     = self.model(interp)
            score     = torch.sigmoid(logit).sum()
            self.model.zero_grad()
            score.backward()
            integrated = integrated + interp.grad.detach()

        # IG = delta × mean_gradient
        attr = (delta * integrated / self.n_steps).squeeze(0)
        attr = attr.mean(dim=0).cpu().numpy()

        # Smooth to reduce high-frequency gradient noise
        from scipy.ndimage import gaussian_filter
        attr = gaussian_filter(attr.astype(np.float32), sigma=3.0)
        return attr.astype(np.float32)



# ─────────────────────────────────────────────────────────────────────
# EXPLAINABILITY ENGINE — main API
# ─────────────────────────────────────────────────────────────────────
class ExplainabilityEngine:
    """
    Unified XAI engine combining Grad-CAM++ and Deep SHAP.

    Parameters
    ----------
    model : nn.Module
        Loaded EfficientNet-B3 classifier (from MammoClassifier.model).
    device : torch.device
    use_shap : bool
        Enable Deep SHAP (slower, ~3-5s per image). Default True.
        Set False for fast Grad-CAM++ only mode.
    """

    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 use_shap: bool = True) -> None:
        self.model    = model
        self.device   = device
        self.use_shap = use_shap

        # Grad-CAM++ on last MBConv block
        self.gradcampp = GradCAMPlusPlus(
            model, model.net.features[7]
        )

        # Integrated Gradients (replaces Deep SHAP)
        self._shap_engine: Optional[DeepSHAPExplainer] = None
        if use_shap:
            try:
                self._shap_engine = DeepSHAPExplainer(model, device)
                method = "captum" if self._shap_engine._use_captum \
                         else "manual IG"
                print(f"  [XAI] Integrated Gradients ready ({method})")
            except Exception as e:
                print(f"  [XAI] IG not available: {e}")
                self.use_shap = False

    def remove_hooks(self):
        self.gradcampp.remove()

    def explain(self,
                crop: np.ndarray,
                ) -> dict:
        """
        Compute both explanations for a single 224×224 uint8 crop.

        Returns
        -------
        dict with keys:
          crop         : original uint8 grayscale crop
          gradcampp    : float32 [0,1] map  (224,224)
          shap         : float32 signed map (224,224)  or None
          overlay_gcpp : BGR uint8 blend for display
          overlay_shap : BGR uint8 SHAP signed colourmap or None
        """
        tensor = _to_tensor(crop, self.device)

        # ── Grad-CAM++ ────────────────────────────────────────────────
        gcpp = self.gradcampp.generate(tensor)

        # ── Deep SHAP ─────────────────────────────────────────────────
        shap_map = None
        if self._shap_engine is not None:
            try:
                shap_map = self._shap_engine.explain(tensor)
            except Exception as e:
                print(f"  [XAI] SHAP failed: {e}")

        # ── Build overlays ────────────────────────────────────────────
        overlay_gcpp = _make_gradcampp_overlay(crop, gcpp)
        overlay_shap = _make_shap_overlay(crop, shap_map) \
                       if shap_map is not None else None

        return dict(
            crop         = crop,
            gradcampp    = gcpp,
            shap         = shap_map,
            overlay_gcpp = overlay_gcpp,
            overlay_shap = overlay_shap,
        )


# ─────────────────────────────────────────────────────────────────────
# OVERLAY BUILDERS
# ─────────────────────────────────────────────────────────────────────
def _make_gradcampp_overlay(crop: np.ndarray,
                             cam: np.ndarray,
                             alpha: float = 0.50) -> np.ndarray:
    """
    Blend Grad-CAM++ heatmap over the grayscale crop.
    Also draws a contour at 0.5 threshold to highlight activation region.
    """
    cam_u8  = np.ascontiguousarray((cam * 255).astype(np.uint8))
    heatmap = cv2.applyColorMap(cam_u8, cv2.COLORMAP_INFERNO)
    bgr     = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    blend   = cv2.addWeighted(bgr, 1 - alpha, heatmap, alpha, 0)

    # Contour at threshold 0.5
    thresh = (cam >= 0.5).astype(np.uint8) * 255
    conts, _ = cv2.findContours(thresh,
                                 cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blend, conts, -1, (0, 255, 200), 1)
    return blend


def _make_shap_overlay(crop: np.ndarray,
                        shap_map: np.ndarray,
                        percentile_clip: float = 99.0) -> np.ndarray:
    """
    Render signed SHAP attributions as a diverging colourmap overlay.
    Red channel = positive SHAP (→ malignant).
    Blue channel = negative SHAP (→ benign).
    """
    # Clip extreme values for display stability
    vmax = np.percentile(np.abs(shap_map), percentile_clip)
    if vmax == 0:
        vmax = 1e-8

    # Separate positive and negative
    pos = np.clip( shap_map / vmax, 0, 1)   # [0,1] → malignant
    neg = np.clip(-shap_map / vmax, 0, 1)   # [0,1] → benign

    bgr = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR).astype(np.float32)

    # Blend: positive → red overlay, negative → blue overlay
    overlay       = bgr.copy()
    overlay[:,:,2] = np.clip(bgr[:,:,2] + pos * 180, 0, 255)  # R
    overlay[:,:,0] = np.clip(bgr[:,:,0] + neg * 180, 0, 255)  # B

    return cv2.addWeighted(bgr, 0.4, overlay, 0.6, 0).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────
# PROFESSIONAL FIGURE
# ─────────────────────────────────────────────────────────────────────
def plot_xai_panel(
    regions_data : List[dict],   # list of explain() results
    region_metas : List[dict],   # list of {label, prob, uncertainty, region_id}
    output_path  : str | Path,
    title        : str = "Explainability — Grad-CAM++ + Deep SHAP",
    dpi          : int = 160,
) -> None:
    """
    Build a professional XAI figure.

    Layout per region (one row):
      [1] Original crop (grayscale)
      [2] Grad-CAM++ overlay + activation contour + colorbar
      [3] SHAP signed attribution + colorbar   (or Grad-CAM++ diff if SHAP N/A)
      [4] Side-by-side intensity profiles (horizontal slice through peak)

    Parameters
    ----------
    regions_data : list of dict from ExplainabilityEngine.explain()
    region_metas : list of dict with label, prob, uncertainty, region_id
    output_path  : path to save the figure
    """
    n = len(regions_data)
    if n == 0:
        print("  [XAI] No regions to plot.")
        return

    has_shap = any(d["shap"] is not None for d in regions_data)
    n_cols   = 4

    fig = plt.figure(
        figsize=(n_cols * 4.5, n * 4.8),
        facecolor='#0d1117'
    )
    fig.suptitle(title,
                 color='white', fontsize=14,
                 fontweight='bold', y=1.002)

    gs = gridspec.GridSpec(
        n, n_cols, figure=fig,
        hspace=0.45, wspace=0.12,
        left=0.04, right=0.96,
        top=0.96, bottom=0.04
    )

    for row, (data, meta) in enumerate(zip(regions_data, region_metas)):
        crop     = data["crop"]
        gcpp     = data["gradcampp"]
        shap_map = data["shap"]
        label    = meta["label"]
        prob     = meta["prob"]
        uncert   = meta.get("uncertainty", 0.0)
        rid      = meta.get("region_id", row + 1)

        color_hex = '#FF4444' if label == 'MALIGNANT' else '#4DA1FF'
        row_title = (f"R{rid:02d} — {label}   "
                     f"P={prob:.3f} ± {uncert:.3f}")

        # ── Col 0: Original crop ──────────────────────────────────────
        ax0 = fig.add_subplot(gs[row, 0])
        ax0.imshow(crop, cmap='gray', vmin=0, vmax=255,
                   interpolation='lanczos')
        ax0.set_title('Original crop', color='#aaaaaa',
                      fontsize=9, pad=4)
        ax0.set_ylabel(row_title, color=color_hex,
                       fontsize=9, labelpad=6)
        ax0.axis('off')
        for sp in ax0.spines.values():
            sp.set_visible(True)
            sp.set_color(color_hex)
            sp.set_linewidth(2.5)

        # ── Col 1: Grad-CAM++ ─────────────────────────────────────────
        ax1 = fig.add_subplot(gs[row, 1])
        im1 = ax1.imshow(crop, cmap='gray', vmin=0, vmax=255,
                         interpolation='lanczos')
        # Overlay Grad-CAM++ with transparency
        im2 = ax1.imshow(gcpp, cmap='inferno', alpha=0.55,
                         vmin=0, vmax=1, interpolation='bilinear')

        # Contour at activation threshold 0.5
        thresh = (gcpp >= 0.5).astype(np.uint8)
        ax1.contour(thresh, levels=[0.5],
                    colors=['#00FFD0'], linewidths=1.2, alpha=0.9)

        # Peak activation marker
        peak_y, peak_x = np.unravel_index(gcpp.argmax(), gcpp.shape)
        ax1.plot(peak_x, peak_y, 'w+', markersize=10, markeredgewidth=1.5)

        ax1.set_title('Grad-CAM++\n(activation localisation)',
                      color='#aaaaaa', fontsize=9, pad=4)
        ax1.axis('off')

        # Colorbar
        div1 = make_axes_locatable(ax1)
        cax1 = div1.append_axes("right", size="4%", pad=0.06)
        cb1  = plt.colorbar(im2, cax=cax1)
        cb1.set_label('Activation', color='white', fontsize=7)
        cb1.ax.yaxis.set_tick_params(color='white', labelcolor='white',
                                      labelsize=7)
        cb1.outline.set_edgecolor('white')

        # ── Col 2: SHAP or Grad-CAM++ diff ───────────────────────────
        ax2 = fig.add_subplot(gs[row, 2])

        if shap_map is not None and shap_map.shape == (224, 224):
            vmax_s = np.percentile(np.abs(shap_map), 99)
            if vmax_s == 0: vmax_s = 1e-8
            norm_s = TwoSlopeNorm(vmin=-vmax_s, vcenter=0, vmax=vmax_s)

            ax2.imshow(crop, cmap='gray', vmin=0, vmax=255,
                       interpolation='lanczos', alpha=0.5)
            im3 = ax2.imshow(shap_map, cmap='bwr', norm=norm_s,
                             alpha=0.65, interpolation='bilinear')
            ax2.set_title('Integrated Gradients\n(RED→malignant  BLUE→benign)',
                          color='#aaaaaa', fontsize=9, pad=4)

            div2 = make_axes_locatable(ax2)
            cax2 = div2.append_axes("right", size="4%", pad=0.06)
            cb2  = plt.colorbar(im3, cax=cax2)
            cb2.set_label('Attribution', color='white', fontsize=7)
            cb2.ax.yaxis.set_tick_params(color='white', labelcolor='white',
                                          labelsize=7)
            cb2.outline.set_edgecolor('white')
        else:
            # Fallback: show Grad-CAM++ in absolute terms
            ax2.imshow(gcpp, cmap='hot', vmin=0, vmax=1,
                       interpolation='bilinear')
            ax2.set_title('Grad-CAM++ (abs)\n[SHAP N/A — pip install shap]',
                          color='#888888', fontsize=8, pad=4)

        ax2.axis('off')

        # ── Col 3: Intensity profiles ─────────────────────────────────
        ax3 = fig.add_subplot(gs[row, 3])
        ax3.set_facecolor('#161b22')

        H, W = gcpp.shape
        # Horizontal profile through peak activation
        peak_y_c = np.clip(peak_y, 0, H - 1)
        xs       = np.arange(W)

        # Normalise crop intensity to [0,1] for comparison
        crop_norm = crop.astype(np.float32) / 255.0

        ax3.plot(xs, crop_norm[peak_y_c, :],
                 color='#aaaaaa', lw=1.2, label='Intensity', alpha=0.8)
        ax3.plot(xs, gcpp[peak_y_c, :],
                 color='#FF8C00', lw=1.8, label='Grad-CAM++')

        if shap_map is not None and shap_map.shape == (224, 224):
            vmax_s   = np.percentile(np.abs(shap_map), 99) + 1e-8
            shap_n   = shap_map[peak_y_c, :] / vmax_s
            # Smooth 1D profile for readability
            from scipy.ndimage import uniform_filter1d
            shap_n_smooth = uniform_filter1d(shap_n, size=9)
            ax3.plot(xs, shap_n_smooth,
                     color='#FF4444', lw=2.0, ls='-',
                     label='IG (norm+smooth)', alpha=0.9)
            ax3.axhline(0, color='#444', lw=0.8)

        ax3.axvline(peak_x, color='white', lw=0.8, ls=':',
                    alpha=0.6, label=f'Peak x={peak_x}')
        ax3.set_xlim(0, W)
        ax3.set_ylim(-1.1, 1.1)
        ax3.set_xlabel('Pixel (horizontal)', color='#888', fontsize=8)
        ax3.set_ylabel('Value (normalised)', color='#888', fontsize=8)
        ax3.set_title(f'Profile — row y={peak_y_c}\n'
                      f'(through peak activation)',
                      color='#aaaaaa', fontsize=9, pad=4)
        ax3.legend(fontsize=7, loc='upper right',
                   facecolor='#1e2530', edgecolor='#444',
                   labelcolor='white')
        ax3.tick_params(colors='white', labelsize=7)
        for sp in ax3.spines.values():
            sp.set_color('#333')

    # ── Footer ────────────────────────────────────────────────────────
    fig.text(
        0.5, -0.01,
        "Grad-CAM++ (Chattopadhay et al. 2018)  ·  "
        "Integrated Gradients (Sundararajan et al. 2017)  ·  "
        "EfficientNet-B3 | AUC=0.7812 | CBIS-DDSM",
        ha='center', color='#555555', fontsize=7
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=dpi,
                bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"  [XAI] Panel saved → {output_path}")


# ─────────────────────────────────────────────────────────────────────
# CONVENIENCE WRAPPER — call directly from pipeline results
# ─────────────────────────────────────────────────────────────────────
def explain_pipeline_results(
    regions      : list,          # list of RegionResult from pipeline.py
    crops        : list,          # list of np.ndarray uint8 crops
    model        : nn.Module,
    device       : torch.device,
    output_path  : str | Path,
    use_shap     : bool = True,
    title        : str  = "",
) -> None:
    """
    Run XAI on all regions from a pipeline result and save the panel.

    Parameters
    ----------
    regions : list of RegionResult (from pipeline.py)
    crops   : list of uint8 numpy arrays, one per region
    model   : clf.model (MammoClassifier.model)
    device  : clf.device
    output_path : where to save the XAI figure
    use_shap : enable Deep SHAP (pip install shap required)
    """
    if not regions:
        print("  [XAI] No regions — nothing to explain.")
        return

    engine = ExplainabilityEngine(model, device, use_shap=use_shap)

    results = []
    metas   = []

    for r, crop in zip(regions, crops):
        print(f"  [XAI] Explaining R{r.region_id:02d} "
              f"({r.label} P={r.prob:.3f})...")
        xai = engine.explain(crop)
        results.append(xai)
        metas.append(dict(
            region_id   = r.region_id,
            label       = r.label,
            prob        = r.prob,
            uncertainty = r.uncertainty,
        ))

    engine.remove_hooks()

    stem  = Path(output_path).stem
    ttl   = title or f"XAI — {stem}"
    plot_xai_panel(results, metas,
                   output_path=output_path,
                   title=ttl)