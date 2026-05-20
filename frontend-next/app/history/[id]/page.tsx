'use client'

import { useState, useEffect } from 'react'
import { useRouter, useParams } from 'next/navigation'
import { t, tBadge, subscribeLang } from '@/lib/i18n'
import { API } from '@/lib/api'
import { fmt } from '@/lib/utils'
import { generateHtmlReport, triggerDownload } from '@/lib/report'
import type { HistoryItem, Region, ToastState } from '@/lib/types'
import { Icon } from '@/components/Icons'
import MammoViewer from '@/components/MammoViewer'
import Toast from '@/components/Toast'

function DetailCell({ label, value, big }: { label: string; value: React.ReactNode; big?: boolean }) {
  return (
    <div style={{ padding: '14px 18px', borderRight: '1px solid var(--brd)', display: 'flex', flexDirection: 'column', gap: 6 }}>
      <div className="clbl">{label}</div>
      <div style={{ minHeight: big ? 28 : 26, display: 'flex', alignItems: 'center' }}>{value}</div>
    </div>
  )
}

function Spinner() {
  return (
    <div style={{ width: 28, height: 28, margin: '0 auto', border: '2.5px solid var(--brd2)', borderTopColor: 'var(--acc)', borderRadius: '50%', animation: 'spin 0.8s linear infinite' }}/>
  )
}

export default function HistoryDetailPage() {
  const router = useRouter()
  const params = useParams()
  const id = params.id as string

  const [, setLang] = useState('')
  useEffect(() => subscribeLang(setLang), [])

  const [item, setItem]       = useState<HistoryItem | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError]     = useState<string | null>(null)
  const [hoveredId, setHoveredId] = useState<number | null>(null)
  const [jsonOpen, setJsonOpen]   = useState(false)
  const [toast, setToast]     = useState<ToastState | null>(null)

  useEffect(() => {
    if (!id) return
    let mounted = true
    setLoading(true); setError(null)
    API.historyDetail(id)
      .then(({ item: it, error: err }) => {
        if (!mounted) return
        setItem(it)
        if (err) setError(err)
      })
      .catch(e => mounted && setError(e instanceof Error ? e.message : String(e)))
      .finally(() => mounted && setLoading(false))
    return () => { mounted = false }
  }, [id])

  if (loading && !item) {
    return (
      <div className="container fade-in" style={{ padding: '80px 32px' }}>
        <div style={{ maxWidth: 520, margin: '0 auto', textAlign: 'center' }}>
          <Spinner/>
          <div style={{ fontSize: 12, color: 'var(--txt3)', marginTop: 14 }}>{t('history.detail.loading')}</div>
        </div>
      </div>
    )
  }

  if (error && !item) {
    return (
      <div className="container fade-in" style={{ padding: '60px 32px' }}>
        <div className="card" style={{ padding: '40px 24px', textAlign: 'center', maxWidth: 520, margin: '0 auto' }}>
          <div style={{ width: 48, height: 48, margin: '0 auto 14px', borderRadius: 12, background: 'var(--bad-bg)', color: 'var(--bad)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}><Icon.Warn s={20}/></div>
          <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 4 }}>{t('history.detail.error.title')}</div>
          <div style={{ fontSize: 12, color: 'var(--txt3)', marginBottom: 18, fontFamily: 'Geist Mono, monospace' }}>{error}</div>
          <button className="btn ghost" onClick={() => router.push('/history')}>{t('history.detail.back')}</button>
        </div>
      </div>
    )
  }

  if (!item) return null

  const report     = (item as any).report || {}
  const regions: Region[] = report.results || []
  const isMal      = item.prediction === 'MALIGNANT'
  const cls        = isMal ? 'suspicious' : item.prediction === 'BENIGN' ? 'benign' : 'normal'
  const overlaySrc = item.has_overlay ? `${API.base()}${item.overlay_url}` : null
  const inferenceS = (report.inference_ms || (item.processing_time || 0) * 1000) / 1000

  const downloadReport = () => {
    const html = generateHtmlReport({
      id:                 item.job_id,
      filename:           item.filename,
      overall_assessment: item.prediction || 'UNKNOWN',
      regions_found:      item.n_regions,
      malignant:          item.malignant,
      benign:             item.benign,
      inference_s:        inferenceS,
      results:            regions,
    })
    triggerDownload(html, `rapport_mammocad_${item.job_id}.html`)
    setToast({ msg: 'Rapport téléchargé', kind: 'success', k: Date.now() })
  }

  return (
    <div className="fade-in">
      <div className="container" style={{ paddingTop: 24, paddingBottom: 0 }}>
        <button className="btn ghost sm" onClick={() => router.push('/history')} style={{ marginBottom: 12 }}>
          {t('history.detail.back')}
        </button>

        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14, flexWrap: 'wrap', gap: 10 }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 4 }}>
              <h1 style={{ fontSize: 20, fontWeight: 700, letterSpacing: '-0.4px' }}>{item.filename}</h1>
              <span className={`badge ${cls}`} style={{ fontSize: 10 }}>{tBadge(item.prediction || 'UNKNOWN')}</span>
            </div>
            <div style={{ fontSize: 11.5, color: 'var(--txt3)', fontFamily: 'Geist Mono, monospace' }}>
              job_id <span style={{ color: 'var(--txt2)' }}>{fmt.shortId(item.job_id)}</span>
              {' · '}{fmt.date(item.timestamp)}
            </div>
          </div>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            <button className="btn primary" onClick={downloadReport}>
              <Icon.Download s={12}/> {t('result.report.download')}
            </button>
          </div>
        </div>

        {/* KPI strip */}
        <div className="card" style={{ padding: 0, overflow: 'hidden', display: 'grid', gridTemplateColumns: '2fr 1fr 1fr 1fr 1fr' }}>
          <DetailCell label={t('result.kpi.assessment')} big value={
            <span className={`badge ${cls}`} style={{ fontSize: 11, padding: '4px 10px' }}>
              {tBadge(report.overall_assessment || item.prediction || 'UNKNOWN')}
            </span>
          }/>
          <DetailCell label={t('result.kpi.regions')} value={<span className="mono" style={{ fontSize: 20, fontWeight: 700, letterSpacing: '-0.5px' }}>{item.n_regions}</span>}/>
          <DetailCell label={t('result.kpi.malignant')} value={<span className="mono" style={{ fontSize: 20, fontWeight: 700, letterSpacing: '-0.5px', color: item.malignant > 0 ? 'var(--bad)' : 'var(--txt)' }}>{item.malignant}</span>}/>
          <DetailCell label={t('result.kpi.benign')} value={<span className="mono" style={{ fontSize: 20, fontWeight: 700, letterSpacing: '-0.5px', color: item.benign > 0 ? 'var(--warn)' : 'var(--txt)' }}>{item.benign}</span>}/>
          <DetailCell label={t('result.kpi.inference')} value={<span className="mono" style={{ fontSize: 16, fontWeight: 600 }}>{fmt.time(inferenceS)}</span>}/>
        </div>
      </div>

      {/* Viewer + side */}
      <div className="container" style={{ marginTop: 14 }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0,1fr) 320px', gap: 14, alignItems: 'start' }}>
          <div className="card" style={{ padding: 14 }}>
            <div className="clbl" style={{ marginBottom: 12 }}>{t('history.detail.overlay')} {item.overlay_url}</div>
            {overlaySrc ? (
              <MammoViewer
                imageSrc={overlaySrc}
                regions={regions}
                hoveredId={hoveredId}
                onHoverChange={setHoveredId}
                showBoxes={true}
                showLabels={false}
              />
            ) : (
              <div style={{ aspectRatio: '4/3', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#000', borderRadius: 10, color: 'var(--txt3)', fontSize: 12 }}>
                {t('history.detail.noOverlay')}
              </div>
            )}
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <div className="card">
              <div className="clbl" style={{ marginBottom: 10 }}>{t('history.detail.confidence')}</div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <span className="mono" style={{ fontSize: 22, fontWeight: 700, letterSpacing: '-0.5px', color: 'var(--txt)' }}>
                  {typeof item.confidence === 'number' ? (item.confidence * 100).toFixed(1) + '%' : '—'}
                </span>
                <div className={`score-bar ${isMal ? 'bad' : 'warn'}`} style={{ flex: 1 }}>
                  <span style={{ width: `${(item.confidence || 0) * 100}%` }}/>
                </div>
              </div>
            </div>

            <div className="card">
              <div className="clbl" style={{ marginBottom: 10 }}>{t('history.detail.metadata')}</div>
              <div style={{ display: 'flex', flexDirection: 'column' }}>
                {[
                  [t('result.meta.filename'),  item.filename],
                  [t('result.meta.jobId'),      item.job_id],
                  [t('result.meta.timestamp'),  fmt.date(item.timestamp)],
                  [t('result.meta.processing'), fmt.time(item.processing_time || 0)],
                  [t('result.meta.hasOverlay'), String(!!item.has_overlay)],
                  [t('result.meta.hasMask'),    String(!!(item as any).has_panel)],
                ].map(([k, v], i, arr) => (
                  <div key={k} style={{ display: 'flex', justifyContent: 'space-between', gap: 10, padding: '7px 0', borderBottom: i < arr.length - 1 ? '1px solid var(--brd)' : 'none' }}>
                    <span style={{ fontSize: 11.5, color: 'var(--txt2)' }}>{k}</span>
                    <span className="mono" style={{ fontSize: 11, color: 'var(--txt)', fontWeight: 500, maxWidth: 170, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{v}</span>
                  </div>
                ))}
              </div>
            </div>

          </div>
        </div>
      </div>

      {/* Rapport de détection */}
      <div className="container" style={{ marginTop: 14, marginBottom: 24 }}>
        <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
          <div style={{ padding: '14px 18px', borderBottom: '1px solid var(--brd)', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 8 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ fontSize: 13, fontWeight: 600 }}>{t('result.report.title')}</span>
              <span className="mono" style={{ fontSize: 10, color: 'var(--txt3)' }}>#{fmt.shortId(item.job_id)}</span>
            </div>
            <button className="btn ghost sm" onClick={downloadReport}><Icon.Download s={11}/> {t('result.report.downloadHtml')}</button>
          </div>

          <div style={{ padding: '18px 18px 6px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 14, padding: '12px 16px', background: 'var(--ab)', borderRadius: 10, border: '1px solid var(--brd)', marginBottom: 16 }}>
              <div>
                <div style={{ fontSize: 9.5, fontWeight: 600, letterSpacing: '0.1em', textTransform: 'uppercase', color: 'var(--txt3)', marginBottom: 4 }}>{t('result.report.assessment')}</div>
                <span className={`badge ${cls}`} style={{ fontSize: 11, padding: '4px 10px' }}>{tBadge(item.prediction || 'UNKNOWN')}</span>
              </div>
              <div style={{ width: 1, height: 36, background: 'var(--brd)' }}/>
              <div style={{ display: 'flex', gap: 20 }}>
                {[
                  { label: t('result.report.detected'),    val: item.n_regions, color: 'var(--txt)' },
                  { label: t('result.report.suspicious'),  val: item.malignant, color: item.malignant > 0 ? 'var(--bad)'  : 'var(--txt3)' },
                  { label: t('result.report.benignCount'), val: item.benign,    color: item.benign   > 0 ? 'var(--warn)' : 'var(--txt3)' },
                ].map(({ label, val, color }) => (
                  <div key={label} style={{ textAlign: 'center' }}>
                    <div className="mono" style={{ fontSize: 20, fontWeight: 700, color, letterSpacing: '-0.5px', lineHeight: 1.1 }}>{val}</div>
                    <div style={{ fontSize: 10, color: 'var(--txt3)', marginTop: 2 }}>{label}</div>
                  </div>
                ))}
              </div>
              <div style={{ marginLeft: 'auto', textAlign: 'right' }}>
                <div style={{ fontSize: 9.5, fontWeight: 600, letterSpacing: '0.1em', textTransform: 'uppercase', color: 'var(--txt3)', marginBottom: 4 }}>{t('result.kpi.inference')}</div>
                <span className="mono" style={{ fontSize: 13, fontWeight: 600 }}>{fmt.time(inferenceS)}</span>
              </div>
            </div>

            {regions.length > 0 ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10, marginBottom: 16 }}>
                {regions.map(r => {
                  const isMalR = r.label === 'MALIGNANT'
                  const [x1, y1, x2, y2] = r.location.map(n => Math.round(n))
                  const barColor = isMalR ? 'var(--bad)' : 'var(--warn)'
                  const bgColor  = isMalR ? 'color-mix(in oklab, var(--bad) 6%, var(--ab))' : 'color-mix(in oklab, var(--warn) 6%, var(--ab))'
                  return (
                    <div key={r.region_id} style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 14, padding: '12px 16px', background: bgColor, borderRadius: 10, border: `1px solid color-mix(in oklab, ${barColor} 20%, var(--brd))` }}>
                      <div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                          <span className="mono" style={{ fontSize: 13, fontWeight: 700 }}>R{r.region_id}</span>
                          <span className={`badge ${r.label.toLowerCase()}`} style={{ fontSize: 9.5, padding: '2px 7px' }}>{tBadge(r.label)}</span>
                        </div>
                        <div style={{ fontSize: 11, color: 'var(--txt3)' }}>
                          {t('result.report.coords')}&nbsp;
                          <span className="mono" style={{ color: 'var(--txt2)', fontSize: 10.5 }}>({x1}, {y1}) → ({x2}, {y2})</span>
                        </div>
                        <div style={{ fontSize: 11, color: 'var(--txt3)', marginTop: 3 }}>
                          {t('result.report.size')}&nbsp;
                          <span className="mono" style={{ color: 'var(--txt2)', fontSize: 10.5 }}>{x2 - x1} × {y2 - y1} px</span>
                        </div>
                      </div>
                      <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', gap: 6 }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <span style={{ fontSize: 10, color: 'var(--txt3)', fontWeight: 600, letterSpacing: '0.08em', textTransform: 'uppercase' }}>{t('result.tip.probability')}</span>
                          <span className="mono" style={{ fontSize: 13, fontWeight: 700, color: barColor }}>{(r.probability * 100).toFixed(1)}%</span>
                        </div>
                        <div style={{ height: 7, borderRadius: 999, background: 'rgba(0,0,0,0.1)', overflow: 'hidden' }}>
                          <div style={{ height: '100%', width: `${r.probability * 100}%`, background: barColor, borderRadius: 999 }}/>
                        </div>
                      </div>
                      <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', gap: 6 }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <span style={{ fontSize: 10, color: 'var(--txt3)', fontWeight: 600, letterSpacing: '0.08em', textTransform: 'uppercase' }}>{t('result.tip.uncertainty')}</span>
                          <span className="mono" style={{ fontSize: 12, fontWeight: 600, color: 'var(--txt2)' }}>±{(r.uncertainty * 100).toFixed(2)}%</span>
                        </div>
                        <div style={{ height: 7, borderRadius: 999, background: 'rgba(0,0,0,0.1)', overflow: 'hidden' }}>
                          <div style={{ height: '100%', width: `${Math.min(1, r.uncertainty * 5) * 100}%`, background: 'var(--ok)', borderRadius: 999 }}/>
                        </div>
                      </div>
                    </div>
                  )
                })}
              </div>
            ) : (
              <div style={{ padding: '20px 0', textAlign: 'center', color: 'var(--txt3)', fontSize: 12, marginBottom: 16 }}>
                {t('result.report.noRegions')}
              </div>
            )}
          </div>

          <button
            onClick={() => setJsonOpen(o => !o)}
            style={{ width: '100%', padding: '10px 18px', border: 'none', borderTop: '1px solid var(--brd)', background: 'transparent', color: 'inherit', display: 'flex', alignItems: 'center', justifyContent: 'space-between', cursor: 'pointer' }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <Icon.Json s={12}/>
              <span style={{ fontSize: 11, color: 'var(--txt3)' }}>{t('result.report.rawJson')}</span>
            </div>
            <span style={{ display: 'inline-block', transform: jsonOpen ? 'rotate(90deg)' : 'rotate(0)', transition: 'transform 0.2s', color: 'var(--txt3)', fontSize: 11 }}>▸</span>
          </button>
          {jsonOpen && (
            <pre className="json-block nice-scroll" style={{ margin: 14 }}>
              {JSON.stringify({ prediction: item.prediction, n_regions: item.n_regions, malignant: item.malignant, benign: item.benign, regions }, null, 2)}
            </pre>
          )}
        </div>
      </div>

      {toast && <Toast key={toast.k} msg={toast.msg} kind={toast.kind} onDone={() => setToast(null)}/>}
    </div>
  )
}
