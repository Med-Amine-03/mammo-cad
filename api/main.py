"""
api/main.py — Mammo-CAD FastAPI backend
========================================

Endpoints:
  POST /predict              → upload image → résultat JSON
  GET  /result/{id}/overlay  → image annotée (bboxes + labels)
  GET  /result/{id}/gradcam  → image Grad-CAM (première région)
  GET  /result/{id}/mask     → masque U-Net
  GET  /result/{id}/report   → rapport JSON complet
  GET  /result/{id}/panel    → panel complet
  GET  /history              → liste de tous les jobs (newest first)
  GET  /history/{id}         → détail d'un job (meta + report)
  GET  /health               → statut des modèles

Usage (depuis project root):
  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Variables d'environnement (optionnelles):
  SEG_CKPT   = checkpoints/unet_best.pth
  CLS_CKPT   = checkpoints/efficientnet_stage2_swa.pth
  RESULTS_DIR = api/results
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# ── project root on path ─────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.inference.classify import MammoClassifier
from src.inference.pipeline import run_pipeline
from src.inference.segment import load_model as load_seg_model

# ─────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────
SEG_CKPT    = Path(os.getenv("SEG_CKPT",  "checkpoints/unet_best.pth"))
CLS_CKPT    = Path(os.getenv("CLS_CKPT",  "checkpoints/efficientnet_inbreast_ft_swa.pth"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "api/results"))
DEVICE      = os.getenv("DEVICE", "auto")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────
# GLOBAL MODEL STATE  (chargé une seule fois au démarrage)
# ─────────────────────────────────────────────────────────────────────
_state: Dict[str, Any] = {
    "seg_model": None,
    "seg_device": None,
    "clf": None,
    "ready": False,
}

_executor = ThreadPoolExecutor(max_workers=1)   # 1 seul worker GPU


# ─────────────────────────────────────────────────────────────────────
# LIFESPAN — charge les modèles une seule fois
# ─────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n[API] Chargement des modèles...")

    seg_path = PROJECT / SEG_CKPT
    cls_path = PROJECT / CLS_CKPT

    if not seg_path.exists():
        raise RuntimeError(f"U-Net checkpoint introuvable : {seg_path}")
    if not cls_path.exists():
        raise RuntimeError(f"EfficientNet checkpoint introuvable : {cls_path}")

    seg_model, seg_device = load_seg_model(str(seg_path), device=DEVICE,
                                           force_reload=True)
    clf = MammoClassifier(str(cls_path), device=DEVICE, tta=16)

    _state["seg_model"]  = seg_model
    _state["seg_device"] = seg_device
    _state["clf"]        = clf
    _state["ready"]      = True

    print(f"[API] U-Net      → {seg_path.name}  (device={seg_device})")
    print(f"[API] Classifier → {cls_path.name}  AUC={clf.val_auc:.4f}  "
          f"thr={clf.threshold:.3f}")
    print("[API] Prêt.\n")

    yield

    _executor.shutdown(wait=False)


# ─────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Mammo-CAD API",
    version="1.0.0",
    description="API de détection des microcalcifications mammaires",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────────────────────────────
class RegionOut(BaseModel):
    region_id  : int
    label      : str
    prob       : float
    uncertainty: float
    bbox       : List[int]


class PredictResponse(BaseModel):
    id             : str
    filename       : str
    prediction     : str          # "MALIGNANT" | "BENIGN"
    confidence     : float        # max prob parmi toutes les régions
    n_regions      : int
    regions        : List[RegionOut]
    processing_time: float
    has_overlay    : bool
    has_gradcam    : bool
    has_mask       : bool


class HistoryItem(BaseModel):
    job_id         : str
    filename       : str
    timestamp      : str          # ISO-8601 UTC
    prediction     : str          # "MALIGNANT" | "BENIGN" | "UNKNOWN"
    n_regions      : int
    malignant      : int
    benign         : int
    confidence     : float
    processing_time: float
    has_overlay    : bool
    has_panel      : bool
    overlay_url    : Optional[str]
    report_url     : str


# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────
def _run_pipeline_sync(job_id: str, img_path: Path) -> dict:
    """Exécute le pipeline dans le thread executor (bloquant)."""
    out_dir = RESULTS_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    report = run_pipeline(
        image_path    = img_path,
        seg_ckpt_path = str(PROJECT / SEG_CKPT),
        cls_ckpt_path = str(PROJECT / CLS_CKPT),
        output_dir    = out_dir,
        gradcam          = True,
        save_crops       = True,
        save_seg_overlay = True,
        preprocess       = True,
        preprocess_clahe = True,
        clf_preloaded    = _state["clf"],
    )
    return report


def _resolve_file(job_id: str, *candidates: str) -> Optional[Path]:
    """Cherche le premier fichier existant parmi les candidats."""
    base = RESULTS_DIR / job_id
    for name in candidates:
        p = base / name
        if p.exists():
            return p
    return None


def _build_response(job_id: str, filename: str, report: dict,
                    elapsed: float) -> PredictResponse:
    # Le rapport utilise "results" avec "probability" et "location"
    regions_raw = report.get("results", report.get("regions", []))
    regions = [
        RegionOut(
            region_id  = r.get("region_id", i),
            label      = r.get("label", "BENIGN"),
            prob       = round(float(r.get("probability", r.get("prob", 0.0))), 4),
            uncertainty= round(float(r.get("uncertainty", 0.0)), 4),
            bbox       = r.get("location", r.get("bbox", [0, 0, 0, 0])),
        )
        for i, r in enumerate(regions_raw)
    ]

    n_mal    = sum(1 for r in regions if r.label == "MALIGNANT")
    max_prob = max((r.prob for r in regions), default=0.0)

    # Vérifier quels fichiers de sortie existent
    # Le pipeline sauvegarde avec le stem de l'image uploadée (ex: "input")
    base = RESULTS_DIR / job_id
    has_overlay = bool(list(base.glob("*_annotated.png")))
    has_mask    = bool(list(base.glob("*_seg_mask.png")))
    crops_dir   = base / "crops"
    has_gradcam = bool(list(crops_dir.glob("*gradcam*"))) if crops_dir.exists() else False

    return PredictResponse(
        id              = job_id,
        filename        = filename,
        prediction      = "MALIGNANT" if n_mal > 0 else "BENIGN",
        confidence      = round(max_prob, 4),
        n_regions       = len(regions),
        regions         = regions,
        processing_time = round(elapsed, 2),
        has_overlay     = has_overlay,
        has_gradcam     = has_gradcam,
        has_mask        = has_mask,
    )


# ─────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    import torch
    return {
        "status"       : "ready" if _state["ready"] else "loading",
        "seg_ckpt"     : SEG_CKPT.name,
        "cls_ckpt"     : CLS_CKPT.name,
        "cls_auc"      : round(_state["clf"].val_auc, 4) if _state["clf"] else None,
        "cls_threshold": round(_state["clf"].threshold, 3) if _state["clf"] else None,
        "device"       : str(_state.get("seg_device", "unknown")),
        "gpu_available": torch.cuda.is_available(),
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    if not _state["ready"]:
        raise HTTPException(503, "Modèles en cours de chargement, réessayez.")

    # Valider le type de fichier
    ext = Path(file.filename).suffix.lower()
    if ext not in {".png", ".jpg", ".jpeg", ".dcm", ".tif", ".tiff"}:
        raise HTTPException(400, f"Format non supporté : {ext}")

    job_id  = str(uuid.uuid4())
    tmp_dir = RESULTS_DIR / job_id
    tmp_dir.mkdir(parents=True, exist_ok=True)
    img_path = tmp_dir / f"input{ext}"

    # Sauvegarder l'image uploadée
    try:
        content = await file.read()
        img_path.write_bytes(content)
    except Exception as e:
        raise HTTPException(500, f"Erreur lecture fichier : {e}")

    # Exécuter le pipeline dans le thread executor (non-bloquant pour l'event loop)
    import asyncio
    import time

    loop = asyncio.get_event_loop()
    t0   = time.perf_counter()
    try:
        report = await loop.run_in_executor(
            _executor, _run_pipeline_sync, job_id, img_path
        )
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(500, f"Erreur pipeline : {e}")
    elapsed = time.perf_counter() - t0

    response = _build_response(job_id, file.filename, report, elapsed)

    # ── Sauvegarder les métadonnées pour l'historique ─────────────────
    meta = {
        "job_id"         : job_id,
        "filename"       : file.filename,
        "timestamp"      : datetime.now(timezone.utc).isoformat(),
        "prediction"     : response.prediction,
        "n_regions"      : response.n_regions,
        "malignant"      : sum(1 for r in response.regions if r.label == "MALIGNANT"),
        "benign"         : sum(1 for r in response.regions if r.label == "BENIGN"),
        "confidence"     : response.confidence,
        "processing_time": response.processing_time,
    }
    (RESULTS_DIR / job_id / "meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    return response


@app.get("/result/{job_id}/overlay")
def get_overlay(job_id: str):
    base = RESULTS_DIR / job_id
    if not base.exists():
        raise HTTPException(404, "Job introuvable")

    # Chercher l'image annotée
    candidates = list(base.glob("*_annotated.png"))
    if not candidates:
        raise HTTPException(404, "Image overlay non disponible")

    return FileResponse(str(candidates[0]), media_type="image/png")


""" @app.get("/result/{job_id}/gradcam")
def get_gradcam(job_id: str):
    crops_dir = RESULTS_DIR / job_id / "crops"
    if not crops_dir.exists():
        raise HTTPException(404, "Job introuvable")

    candidates = sorted(crops_dir.glob("*gradcam*.png"))
    if not candidates:
        raise HTTPException(404, "Grad-CAM non disponible")

    return FileResponse(str(candidates[0]), media_type="image/png") """


@app.get("/result/{job_id}/mask")
def get_mask(job_id: str):
    base = RESULTS_DIR / job_id
    if not base.exists():
        raise HTTPException(404, "Job introuvable")

    candidates = list(base.glob("*_seg_mask.png"))
    if not candidates:
        raise HTTPException(404, "Masque non disponible")

    return FileResponse(str(candidates[0]), media_type="image/png")


@app.get("/result/{job_id}/report")
def get_report(job_id: str):
    base = RESULTS_DIR / job_id
    if not base.exists():
        raise HTTPException(404, "Job introuvable")

    # Chercher spécifiquement le rapport pipeline (pas meta.json)
    candidates = list(base.glob("*_report.json"))
    if not candidates:
        raise HTTPException(404, "Rapport JSON non disponible")

    with open(candidates[0]) as f:
        data = json.load(f)
    return JSONResponse(data)


@app.get("/result/{job_id}/panel")
def get_panel(job_id: str):
    base = RESULTS_DIR / job_id
    if not base.exists():
        raise HTTPException(404, "Job introuvable")

    candidates = list(base.glob("*_panel.png"))
    if not candidates:
        raise HTTPException(404, "Panel non disponible")

    return FileResponse(str(candidates[0]), media_type="image/png")


# ─────────────────────────────────────────────────────────────────────
# HISTORY HELPERS
# ─────────────────────────────────────────────────────────────────────

def _job_timestamp(job_dir: Path) -> str:
    """Retourne le timestamp ISO-8601 d'un job.
    Priorité : meta.json → ctime de input_report.json → ctime du dossier.
    """
    meta_file = job_dir / "meta.json"
    if meta_file.exists():
        try:
            m = json.loads(meta_file.read_text(encoding="utf-8"))
            if "timestamp" in m:
                return m["timestamp"]
        except Exception:
            pass

    # Fallback : ctime du rapport JSON
    for report_file in job_dir.glob("*_report.json"):
        ts = report_file.stat().st_ctime
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

    # Fallback ultime : ctime du dossier
    ts = job_dir.stat().st_ctime
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _build_history_item(job_dir: Path) -> Optional[HistoryItem]:
    """Construit un HistoryItem depuis un dossier de job.
    Retourne None si le dossier n'est pas un job valide.
    """
    job_id = job_dir.name

    # ── Lire meta.json (jobs récents) ────────────────────────────────
    meta_file = job_dir / "meta.json"
    if meta_file.exists():
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    else:
        meta = {}

    # ── Lire le rapport pipeline (fallback pour anciens jobs) ────────
    report_candidates = list(job_dir.glob("*_report.json"))
    report: Dict[str, Any] = {}
    if report_candidates:
        try:
            report = json.loads(report_candidates[0].read_text(encoding="utf-8"))
        except Exception:
            pass

    # Un job valide doit avoir au moins le rapport
    if not report and not meta:
        return None

    # ── Récupérer les champs ─────────────────────────────────────────
    filename = meta.get("filename") or "unknown"
    if filename == "unknown" and report:
        # Essayer de deviner depuis le champ "image" du rapport
        img_field = report.get("image", "")
        if img_field:
            filename = Path(img_field).name

    timestamp = _job_timestamp(job_dir)

    n_regions = int(meta.get("n_regions", report.get("regions_found", 0)))
    malignant = int(meta.get("malignant",  report.get("malignant",  0)))
    benign    = int(meta.get("benign",      report.get("benign",    0)))

    # Prédiction : meta.json si présent, sinon dériver du compte malignant
    if "prediction" in meta:
        prediction = meta["prediction"]
        if prediction not in ("MALIGNANT", "BENIGN"):
            prediction = "MALIGNANT" if malignant > 0 else "BENIGN"
    else:
        prediction = "MALIGNANT" if malignant > 0 else "BENIGN"

    # Confidence : meta.json si présent, sinon max(probability) parmi les résultats
    if "confidence" in meta and float(meta["confidence"]) > 0:
        confidence = float(meta["confidence"])
    else:
        results = report.get("results", [])
        if results:
            confidence = max(
                float(r.get("probability", r.get("prob", 0.0)))
                for r in results
            )
        else:
            confidence = 0.0

    processing_time = float(meta.get("processing_time", report.get("inference_ms", 0) / 1000))

    # ── Fichiers de sortie ───────────────────────────────────────────
    has_overlay = bool(list(job_dir.glob("*_annotated.png")))
    has_panel   = bool(list(job_dir.glob("*_panel.png")))
    overlay_url = f"/result/{job_id}/overlay" if has_overlay else None
    report_url  = f"/result/{job_id}/report"

    return HistoryItem(
        job_id          = job_id,
        filename        = filename,
        timestamp       = timestamp,
        prediction      = prediction,
        n_regions       = n_regions,
        malignant       = malignant,
        benign          = benign,
        confidence      = round(confidence, 4),
        processing_time = round(processing_time, 2),
        has_overlay     = has_overlay,
        has_panel       = has_panel,
        overlay_url     = overlay_url,
        report_url      = report_url,
    )


# ─────────────────────────────────────────────────────────────────────
# HISTORY ROUTES
# ─────────────────────────────────────────────────────────────────────

@app.get("/history", response_model=List[HistoryItem])
def get_history():
    """Retourne la liste de tous les jobs, triés du plus récent au plus ancien."""
    items: List[HistoryItem] = []

    if not RESULTS_DIR.exists():
        return []

    for job_dir in RESULTS_DIR.iterdir():
        if not job_dir.is_dir():
            continue
        item = _build_history_item(job_dir)
        if item is not None:
            items.append(item)

    # Trier newest first
    items.sort(key=lambda x: x.timestamp, reverse=True)
    return items


@app.get("/history/{job_id}")
def get_history_detail(job_id: str):
    """Retourne le détail complet d'un job : meta + rapport pipeline."""
    job_dir = RESULTS_DIR / job_id
    if not job_dir.exists() or not job_dir.is_dir():
        raise HTTPException(404, "Job introuvable")

    item = _build_history_item(job_dir)
    if item is None:
        raise HTTPException(404, "Job invalide ou incomplet")

    # Ajouter le rapport pipeline complet
    report_candidates = list(job_dir.glob("*_report.json"))
    full_report: Dict[str, Any] = {}
    if report_candidates:
        try:
            full_report = json.loads(report_candidates[0].read_text(encoding="utf-8"))
        except Exception:
            pass

    return JSONResponse({
        **item.model_dump(),
        "report": full_report,
    })
