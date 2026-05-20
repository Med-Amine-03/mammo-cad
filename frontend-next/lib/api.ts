import type { PredictResult, HistoryItem, HealthStatus, Region } from './types'
import { ApiSettings, HistoryStore } from './storage'
import { fileToDataUrl, imageDims, sleep, mulberry } from './utils'
import { tBadge } from './i18n'

// ── URL helpers ───────────────────────────────────────────────────────────
export const API = {
  base() { return ApiSettings.getBase() },
  forceMock() { return ApiSettings.getForceMock() },
  overlayUrl(id: string) { return `${API.base()}/result/${id}/overlay` },
  maskUrl(id: string)    { return `${API.base()}/result/${id}/mask` },
  gradcamUrl(id: string) { return `${API.base()}/result/${id}/gradcam` },
  reportUrl(id: string)  { return `${API.base()}/result/${id}/report` },

  async health(signal?: AbortSignal): Promise<HealthStatus> {
    if (API.forceMock()) return { status: 'ready', device: 'mock', gpu_available: false, source: 'mock' }
    try {
      const res = await fetch(`${API.base()}/health`, { method: 'GET', cache: 'no-store', signal })
      if (!res.ok) throw new Error(`/health ${res.status}`)
      const j = await res.json()
      return { ...j, source: 'live' }
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e)
      return { status: 'offline', error: msg, source: 'offline' }
    }
  },

  async predict(file: File, signal?: AbortSignal): Promise<PredictResult> {
    if (API.forceMock()) return mockPredict(file)
    const fd = new FormData()
    fd.append('file', file, file.name)
    try {
      const res = await fetch(`${API.base()}/predict`, { method: 'POST', body: fd, signal })
      if (!res.ok) {
        let detail = ''
        try { detail = (await res.json()).detail } catch {}
        throw new Error(detail || `Prediction failed (${res.status})`)
      }
      const j = await res.json()
      return normalizePrediction(j, file)
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e)
      console.warn('[predict] live call failed, using mock:', msg)
      const mock = await mockPredict(file)
      ;(mock as PredictResult & { _mocked_due_to_error?: string })._mocked_due_to_error = msg
      return mock
    }
  },

  async history(signal?: AbortSignal): Promise<{ source: string; items: HistoryItem[]; error?: string }> {
    if (API.forceMock()) {
      return { source: 'mock', items: HistoryStore.list().map(e => toServerShape(e)) }
    }
    try {
      const res = await fetch(`${API.base()}/history`, { method: 'GET', cache: 'no-store', signal })
      if (!res.ok) throw new Error(`/history ${res.status}`)
      const data = await res.json()
      const items: HistoryItem[] = Array.isArray(data) ? data : (data.items || [])
      return { source: 'live', items }
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e)
      return { source: 'offline', error: msg, items: HistoryStore.list().map(e => toServerShape(e)) }
    }
  },

  async historyDetail(id: string, signal?: AbortSignal): Promise<{ source: string; item: HistoryItem; error?: string }> {
    if (API.forceMock()) {
      const local = HistoryStore.get(id)
      if (!local) throw new Error('Not found')
      return { source: 'mock', item: toServerShape(local, true) }
    }
    try {
      const res = await fetch(`${API.base()}/history/${id}`, { method: 'GET', cache: 'no-store', signal })
      if (!res.ok) throw new Error(`/history/${id} ${res.status}`)
      const item = await res.json()
      return { source: 'live', item }
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e)
      const local = HistoryStore.get(id)
      if (local) return { source: 'offline', error: msg, item: toServerShape(local, true) }
      throw e
    }
  },
}

// ── Normalize backend response ────────────────────────────────────────────
function normalizePrediction(raw: Record<string, unknown>, file: File): PredictResult {
  const rawRegions = (raw.results || raw.regions || []) as Record<string, unknown>[]
  const regions: Region[] = rawRegions.map((r, i) => ({
    region_id: (r.region_id as number) ?? i + 1,
    location: ((r.location || r.bbox || [0, 0, 0, 0]) as [number, number, number, number]),
    label: String(r.label || '').toUpperCase(),
    probability: (r.probability as number) ?? (r.prob as number) ?? 0,
    uncertainty: (r.uncertainty as number) ?? 0,
  }))

  const malignant = regions.filter(r => r.label === 'MALIGNANT').length
  const benign    = regions.filter(r => r.label === 'BENIGN').length

  let assessment = (raw.overall_assessment || raw.assessment) as string | undefined
  if (!assessment) {
    if (malignant >= 1) assessment = 'SUSPICIOUS'
    else if (benign > 0) assessment = 'BENIGN'
    else assessment = 'NORMAL'
  }

  return {
    id: (raw.id as string) || `r_${Date.now().toString(36)}`,
    filename: (raw.filename as string) || file?.name || 'mammogram.png',
    overall_assessment: assessment,
    regions_found: (raw.regions_found as number) ?? regions.length,
    malignant,
    benign,
    inference_s: (raw.inference_s as number) ?? (raw.processing_time as number) ?? 0,
    has_overlay: (raw.has_overlay as boolean) ?? false,
    has_mask: (raw.has_mask as boolean) ?? false,
    has_gradcam: (raw.has_gradcam as boolean) ?? false,
    image_size: (raw.image_size as [number, number]) || null,
    results: regions,
    _raw: raw,
  }
}

// ── Convert local entry to server shape ───────────────────────────────────
import type { LocalHistoryEntry } from './storage'

function toServerShape(entry: LocalHistoryEntry, withReport = false): HistoryItem {
  const r = entry._result || {} as Partial<PredictResult>
  const base: HistoryItem = {
    job_id: entry.id,
    filename: entry.filename,
    timestamp: new Date(entry.date).toISOString(),
    prediction: entry.overall_assessment === 'SUSPICIOUS' ? 'MALIGNANT'
              : entry.overall_assessment === 'BENIGN' ? 'BENIGN' : 'UNKNOWN',
    n_regions: entry.regions_found || 0,
    malignant: entry.malignant || 0,
    benign: entry.benign || 0,
    confidence: (r.results || []).reduce((m, x) => Math.max(m, x.probability || 0), 0),
    processing_time: entry.inference_s || 0,
    has_overlay: !!(r.has_overlay),
    has_panel: false,
    overlay_url: `/result/${entry.id}/overlay`,
    report_url: `/result/${entry.id}/report`,
  }
  if (!withReport) return base
  return {
    ...base,
    report: {
      regions_found: r.regions_found || 0,
      inference_ms: Math.round((entry.inference_s || 0) * 1000),
      overall_assessment: r.overall_assessment || '',
      results: r.results || [],
    },
  }
}

// ── Mock prediction ───────────────────────────────────────────────────────
async function mockPredict(file: File): Promise<PredictResult> {
  const dataUrl = await fileToDataUrl(file)
  const dims = await imageDims(dataUrl)
  const W = dims.w, H = dims.h
  const seed = (file?.name || 'demo').length + (file?.size || 0)
  const rand = mulberry(seed)
  const n = 4 + Math.floor(rand() * 3)
  const results: Region[] = []
  let mal = 0, ben = 0
  for (let i = 0; i < n; i++) {
    const cx = 0.18 + rand() * 0.55
    const cy = 0.22 + rand() * 0.55
    const sz = 0.018 + rand() * 0.04
    const ar = 0.85 + rand() * 0.4
    const w = sz * ar, h = sz / ar
    const x1 = Math.round((cx - w / 2) * W)
    const y1 = Math.round((cy - h / 2) * H)
    const x2 = Math.round((cx + w / 2) * W)
    const y2 = Math.round((cy + h / 2) * H)
    const isMal = i === 0 || rand() > 0.75
    if (isMal) mal++; else ben++
    results.push({
      region_id: i + 1,
      location: [x1, y1, x2, y2],
      label: isMal ? 'MALIGNANT' : 'BENIGN',
      probability: isMal ? 0.55 + rand() * 0.35 : 0.18 + rand() * 0.35,
      uncertainty: 0.03 + rand() * 0.10,
    })
  }
  results.sort((a, b) => b.probability - a.probability)
  await sleep(900 + Math.floor(rand() * 1200))
  return {
    id: `mock_${Date.now().toString(36)}`,
    filename: file?.name || 'sample.png',
    overall_assessment: mal >= 1 ? 'SUSPICIOUS' : 'BENIGN',
    regions_found: results.length,
    malignant: mal,
    benign: ben,
    inference_s: 1.4 + rand() * 0.9,
    has_overlay: false,
    has_mask: false,
    has_gradcam: false,
    image_size: [W, H],
    results,
    _mocked: true,
    _dataUrl: dataUrl,
  }
}
