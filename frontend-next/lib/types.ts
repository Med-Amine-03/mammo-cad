export interface Region {
  region_id: number
  location: [number, number, number, number]
  label: 'MALIGNANT' | 'BENIGN' | string
  probability: number
  uncertainty: number
}

export interface PredictResult {
  id: string
  filename: string
  overall_assessment: string
  regions_found: number
  malignant: number
  benign: number
  inference_s: number
  has_overlay: boolean
  has_mask: boolean
  has_gradcam: boolean
  image_size: [number, number] | null
  results: Region[]
  _mocked?: boolean
  _dataUrl?: string
  _raw?: unknown
}

export interface HistoryItem {
  job_id: string
  filename: string
  timestamp: string
  prediction: 'MALIGNANT' | 'BENIGN' | 'UNKNOWN'
  n_regions: number
  malignant: number
  benign: number
  confidence: number
  processing_time: number
  has_overlay: boolean
  has_panel: boolean
  overlay_url: string | null
  report_url: string
  report?: PipelineReport
}

export interface PipelineReport {
  regions_found: number
  inference_ms: number
  overall_assessment: string
  results: Region[]
}

export interface HealthStatus {
  status: 'ready' | 'loading' | 'offline' | string
  seg_ckpt?: string
  cls_ckpt?: string
  cls_auc?: number
  cls_threshold?: number
  device?: string
  gpu_available?: boolean
  source?: 'live' | 'offline' | 'mock'
  error?: string
}

export type ToastKind = 'info' | 'success' | 'warn' | 'error'

export interface ToastState {
  msg: string
  kind: ToastKind
  k: number
}

export type Lang = 'fr' | 'en'
