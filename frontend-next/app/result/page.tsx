'use client'

import { useState, useEffect, useMemo, useCallback } from 'react'
import { useRouter } from 'next/navigation'
import { t, tBadge, subscribeLang } from '@/lib/i18n'
import { LastResult } from '@/lib/storage'
import { API } from '@/lib/api'
import { fmt, highlightJson } from '@/lib/utils'
import { generateHtmlReport, triggerDownload } from '@/lib/report'
import type { PredictResult, Region, ToastState } from '@/lib/types'
import { Icon } from '@/components/Icons'
import MammoViewer from '@/components/MammoViewer'
import Toast from '@/components/Toast'

function SummaryCell({ label, value, big }: { label: string; value: React.ReactNode; big?: boolean }) {
  return (
    <div style={{ padding: '14px 18px', borderRight: '1px solid var(--brd)', display: 'flex', flexDirection: 'column', gap: 6 }}>
      <div className="clbl">{label}</div>
      <div style={{ minHeight: big ? 28 : 26, display: 'flex', alignItems: 'center' }}>{value}</div>
    </div>
  )
}

function Metric({ label, value, bar, barClass }: { label: string; value: string; bar?: number; barClass?: string }) {
  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
        <span style={{ fontSize: 9.5, fontWeight: 600, letterSpacing: '0.1em', textTransform: 'uppercase', color: 'var(--txt3)' }}>{label}</span>
        <span className="mono" style={{ fontSize: 11, color: 'var(--txt)', fontWeight: 600 }}>{value}</span>
      </div>
      {typeof bar === 'number' && (
        <div className={`score-bar${barClass ? ' ' + barClass : ''}`}>
          <span style={{ width: `${Math.max(0, Math.min(1, bar)) * 100}%` }}/>
        </div>
      )}
    </div>
  )
}

function EndpointRow({ method, path, status, link }: { method: string; path: string; status?: string; link?: string | null }) {
  const colors: Record<string, { bg: string; fg: string }> = {
    GET:  { bg: 'var(--info-bg)', fg: 'var(--info)' },
    POST: { bg: 'var(--ok-bg)',   fg: 'var(--ok)' },
  }
  const c = colors[method] || { bg: 'var(--ab)', fg: 'var(--txt2)' }
  const Wrap = link ? 'a' : 'div'
  return (
    <Wrap
      {...(link ? { href: link, target: '_blank', rel: 'noopener noreferrer' } : {})}
      style={{
        display: 'flex', alignItems: 'center', gap: 8,
        padding: '5px 8px', background: 'var(--ab)',
        border: '1px solid var(--brd)', borderRadius: 6,
        cursor: link ? 'pointer' : 'default', transition: 'border-color 0.15s',
        textDecoration: 'none',
      } as React.CSSProperties}
    >
      <span style={{ fontSize: 9, fontWeight: 700, padding: '2px 6px', borderRadius: 4, background: c.bg, color: c.fg, flexShrink: 0 }}>
        {method}
      </span>
      <span style={{ color: 'var(--txt2)', flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{path}</span>
      {status && <span style={{ fontSize: 9, color: 'var(--ok)', fontWeight: 600 }}>{status}</span>}
      {link && <span style={{ color: 'var(--txt3)' }}><Icon.Link s={10}/></span>}
    </Wrap>
  )
}

export default function ResultPage() {
  const router = useRouter()
  const [, setLang] = useState('')
  useEffect(() => subscribeLang(setLang), [])

  const [result, setResult]     = useState<PredictResult | null>(null)
  const [dataUrl, setDataUrl]   = useState<string | null>(null)
  const [hoveredId, setHoveredId]   = useState<number | null>(null)
  const [selectedId, setSelectedId] = useState<number | null>(null)
  const [jsonOpen, setJsonOpen]     = useState(false)
  const [toast, setToast]           = useState<ToastState | null>(null)

  useEffect(() => {
    const cached = LastResult.load()
    if (cached?.result) {
      setResult(cached.result)
      setDataUrl(cached.dataUrl)
    }
  }, [])

  const showToast = (msg: string, kind: ToastState['kind'] = 'info') =>
    setToast({ msg, kind, k: Date.now() })

  const activeRegion = useMemo(() => {
    const r = result?.results || []
    if (selectedId != null) return r.find(x => x.region_id === selectedId)
    if (hoveredId != null)  return r.find(x => x.region_id === hoveredId)
    return r.find(x => x.label === 'MALIGNANT') || r[0]
  }, [result, selectedId, hoveredId])

  const downloadReport = useCallback(() => {
    if (!result) return
    const html = generateHtmlReport({
      id:                 result.id,
      filename:           result.filename,
      overall_assessment: result.overall_assessment,
      regions_found:      result.regions_found,
      malignant:          result.malignant,
      benign:             result.benign,
      inference_s:        result.inference_s,
      results:            result.results || [],
      image_size:         result.image_size,
    })
    triggerDownload(html, `rapport_mammocad_${result.id}.html`)
    showToast('Rapport téléchargé', 'success')
  }, [result])

  const copyJson = async () => {
    if (!result) return
    try {
      const payload = {
        overall_assessment: result.overall_assessment,
        regions_found: result.regions_found,
        malignant: result.malignant,
        benign: result.benign,
        inference_s: result.inference_s,
        results: result.results,
      }
      await navigator.clipboard.writeText(JSON.stringify(payload, null, 2))
      showToast(t('result.actions.copied'), 'success')
    } catch { showToast('Copy failed', 'error') }
  }

  if (!result) {
    return (
      <div className="container fade-in" style={{ padding: '80px 32px' }}>
        <div className="card" style={{ textAlign: 'center', padding: '48px 24px' }}>
          <div style={{ width: 48, height: 48, margin: '0 auto 16px', borderRadius: 12, background: 'var(--ab)', border: '1px solid var(--brd)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--txt3)' }}>
            <Icon.Json s={20}/>
          </div>
          <div style={{ fontSize: 15, fontWeight: 600, marginBottom: 6 }}>{t('result.empty.title')}</div>
          <div style={{ fontSize: 12, color: 'var(--txt3)', marginBottom: 18 }}>{t('result.empty.desc')}</div>
          <button className="btn primary" onClick={() => router.push('/')}>
            {t('result.empty.cta')} <Icon.ArrowRight s={12}/>
          </button>
        </div>
      </div>
    )
  }

  const imageSrc = (result.has_overlay && !(result as any)._mocked)
    ? API.overlayUrl(result.id)
    : dataUrl || undefined

  const assessmentClass = (() => {
    const a = (result.overall_assessment || '').toUpperCase()
    if (a === 'SUSPICIOUS' || a === 'MALIGNANT') return 'suspicious'
    if (a === 'BORDERLINE') return 'borderline'
    if (a === 'BENIGN') return 'benign'
    return 'normal'
  })()

  return (
    <div className="fade-in">
      {/* Summary header */}
      <div className="container" style={{ paddingTop: 28, paddingBottom: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14, flexWrap: 'wrap', gap: 10 }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 4 }}>
              <h1 style={{ fontSize: 20, fontWeight: 700, letterSpacing: '-0.4px' }}>{t('result.title')}</h1>
              {(result as any)._mocked && (
                <span className="badge muted" title={t('result.demo')}>
                  <Icon.Warn s={10}/> DEMO
                </span>
              )}
            </div>
            <div style={{ fontSize: 11.5, color: 'var(--txt3)', fontFamily: 'Geist Mono, monospace' }}>
              {result.filename} · id <span style={{ color: 'var(--txt2)' }}>{fmt.shortId(result.id)}</span>
            </div>
          </div>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            <button className="btn ghost" onClick={() => router.push('/')}>
              <Icon.Reset s={12}/> {t('result.newAnalysis')}
            </button>
            <button className="btn primary" onClick={downloadReport}>
              <Icon.Download s={12}/> {t('result.report.download')}
            </button>
          </div>
        </div>

        {/* KPI strip */}
        <div className="card" style={{ padding: 0, overflow: 'hidden', display: 'grid', gridTemplateColumns: '2fr 1fr 1fr 1fr 1fr' }}>
          <SummaryCell label={t('result.kpi.assessment')} big value={
            <span className={`badge ${assessmentClass}`} style={{ fontSize: 11, padding: '4px 10px' }}>{tBadge(result.overall_assessment)}</span>
          }/>
          <SummaryCell label={t('result.kpi.regions')} value={<span className="mono" style={{ fontSize: 20, fontWeight: 700, letterSpacing: '-0.5px' }}>{result.regions_found}</span>}/>
          <SummaryCell label={t('result.kpi.malignant')} value={<span className="mono" style={{ fontSize: 20, fontWeight: 700, letterSpacing: '-0.5px', color: result.malignant > 0 ? 'var(--bad)' : 'var(--txt)' }}>{result.malignant}</span>}/>
          <SummaryCell label={t('result.kpi.benign')} value={<span className="mono" style={{ fontSize: 20, fontWeight: 700, letterSpacing: '-0.5px', color: result.benign > 0 ? 'var(--warn)' : 'var(--txt)' }}>{result.benign}</span>}/>
          <SummaryCell label={t('result.kpi.inference')} value={<span className="mono" style={{ fontSize: 16, fontWeight: 600 }}>{fmt.time(result.inference_s)}</span>}/>
        </div>
      </div>

      {/* Viewer + side panel */}
      <div className="container" style={{ marginTop: 14 }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) 320px', gap: 14, alignItems: 'start' }}>
          <div className="card" style={{ padding: 14 }}>
            <div className="clbl" style={{ marginBottom: 12 }}>{t('result.viewer.title')}</div>

            {imageSrc ? (
              <MammoViewer
                imageSrc={imageSrc}
                regions={result.results || []}
                imageSize={result.image_size}
                hoveredId={hoveredId ?? selectedId}
                onHoverChange={setHoveredId}
                showBoxes={true}
                showLabels={true}
              />
            ) : (
              <div style={{ aspectRatio: '4/3', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#000', borderRadius: 10, color: 'var(--txt3)', fontSize: 12 }}>
                Image preview unavailable (DICOM with no rendered pixels)
              </div>
            )}

            <div style={{ display: 'flex', alignItems: 'center', gap: 14, marginTop: 10, fontSize: 11, color: 'var(--txt3)' }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <span style={{ width: 10, height: 10, borderRadius: 2, background: '#f87171' }}/>
                {tBadge('MALIGNANT')}
              </span>
              <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <span style={{ width: 10, height: 10, borderRadius: 2, background: '#fbbf24' }}/>
                {tBadge('BENIGN')}
              </span>
              <span style={{ marginLeft: 'auto' }}>{t('result.viewer.hoverHint')}</span>
            </div>
          </div>

          {/* Side panel */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <div className="card">
              <div className="clbl" style={{ marginBottom: 10 }}>
                {activeRegion ? `${t('result.tip.region')} R${activeRegion.region_id}` : t('result.tip.region')}
              </div>
              {activeRegion ? (
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 9, marginBottom: 14 }}>
                    <span className={`badge ${activeRegion.label.toLowerCase()}`} style={{ fontSize: 10, padding: '4px 9px' }}>
                      {tBadge(activeRegion.label)}
                    </span>
                    <span className="mono" style={{ fontSize: 13, fontWeight: 600, color: 'var(--txt)' }}>
                      {fmt.pct(activeRegion.probability, 2)}
                    </span>
                  </div>
                  <div style={{ display: 'grid', gap: 10 }}>
                    <Metric label={t('result.tip.probability')} value={fmt.prob(activeRegion.probability)} bar={activeRegion.probability} barClass={activeRegion.label === 'MALIGNANT' ? 'bad' : 'warn'}/>
                    <Metric label={t('result.tip.uncertainty')} value={fmt.unc(activeRegion.uncertainty)} bar={Math.min(1, activeRegion.uncertainty * 5)} barClass="ok"/>
                    <div>
                      <div style={{ fontSize: 9.5, fontWeight: 600, letterSpacing: '0.1em', textTransform: 'uppercase', color: 'var(--txt3)', marginBottom: 4 }}>{t('result.tip.bbox')}</div>
                      <div className="mono" style={{ fontSize: 11, color: 'var(--txt2)', background: 'var(--ab)', padding: '6px 9px', borderRadius: 6, border: '1px solid var(--brd)' }}>
                        {fmt.bbox(activeRegion.location)}
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                <div style={{ fontSize: 11.5, color: 'var(--txt3)' }}>{t('result.noRegions')}</div>
              )}
            </div>

          </div>
        </div>
      </div>

      {/* Region table */}
      <div className="container" style={{ marginTop: 14 }}>
        <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
          <div style={{ padding: '14px 18px', borderBottom: '1px solid var(--brd)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div className="clbl">{t('result.table.title')} · {result.results?.length || 0} {t('result.table.rows')}</div>
            <span style={{ fontSize: 10.5, color: 'var(--txt3)' }}>{t('result.table.hoverHint')}</span>
          </div>
          <table className="table">
            <thead>
              <tr>
                <th style={{ width: 60 }}>{t('result.table.region')}</th>
                <th>{t('result.table.bbox')} <span style={{ textTransform: 'none', color: 'var(--txt3)', fontWeight: 400, letterSpacing: 0, marginLeft: 4 }}>[x1, y1, x2, y2]</span></th>
                <th style={{ width: 130 }}>{t('result.table.label')}</th>
                <th style={{ width: 140 }}>{t('result.table.probability')}</th>
                <th style={{ width: 120 }}>{t('result.table.uncertainty')}</th>
                <th style={{ width: 50 }}/>
              </tr>
            </thead>
            <tbody>
              {(result.results || []).map(r => {
                const isActive = (selectedId ?? hoveredId) === r.region_id
                return (
                  <tr
                    key={r.region_id}
                    onMouseEnter={() => setHoveredId(r.region_id)}
                    onMouseLeave={() => setHoveredId(null)}
                    onClick={() => setSelectedId(s => s === r.region_id ? null : r.region_id)}
                    style={isActive ? { background: 'var(--ab)' } : undefined}
                  >
                    <td className="mono" style={{ fontWeight: 600, color: 'var(--txt)' }}>R{r.region_id}</td>
                    <td className="mono" style={{ fontSize: 11, color: 'var(--txt2)' }}>{fmt.bbox(r.location)}</td>
                    <td><span className={`badge ${r.label.toLowerCase()}`}>{tBadge(r.label)}</span></td>
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 9 }}>
                        <div className={`score-bar ${r.label === 'MALIGNANT' ? 'bad' : 'warn'}`} style={{ flex: 1, maxWidth: 120 }}>
                          <span style={{ width: `${r.probability * 100}%` }}/>
                        </div>
                        <span className="mono" style={{ fontSize: 11.5, minWidth: 54, textAlign: 'right', fontWeight: 600 }}>
                          {r.probability.toFixed(3)}
                        </span>
                      </div>
                    </td>
                    <td className="mono" style={{ color: 'var(--txt2)', fontSize: 11.5 }}>{fmt.unc(r.uncertainty)}</td>
                    <td style={{ textAlign: 'right' }}>
                      <span style={{ display: 'inline-block', width: 6, height: 6, borderRadius: '50%', background: isActive ? (r.label === 'MALIGNANT' ? 'var(--bad)' : 'var(--warn)') : 'transparent', transition: 'background 0.15s' }}/>
                    </td>
                  </tr>
                )
              })}
              {(result.results || []).length === 0 && (
                <tr><td colSpan={6} style={{ textAlign: 'center', color: 'var(--txt3)', padding: 28 }}>{t('result.table.noRegions')}</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Human-readable report */}
      <div className="container" style={{ marginTop: 14, marginBottom: 20 }}>
        <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
          {/* Report header */}
          <div style={{ padding: '14px 18px', borderBottom: '1px solid var(--brd)', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 8 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ fontSize: 13, fontWeight: 600 }}>{t('result.report.title')}</span>
              <span className="mono" style={{ fontSize: 10, color: 'var(--txt3)' }}>#{fmt.shortId(result.id)}</span>
            </div>
            <div style={{ display: 'flex', gap: 6 }}>
              <button className="btn ghost sm" onClick={copyJson}><Icon.Json s={11}/> {t('result.report.copyJson')}</button>
              <button className="btn ghost sm" onClick={downloadReport}><Icon.Download s={11}/> {t('result.report.downloadHtml')}</button>
            </div>
          </div>

          <div style={{ padding: '18px 18px 6px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 14, padding: '12px 16px', background: 'var(--ab)', borderRadius: 10, border: '1px solid var(--brd)', marginBottom: 16 }}>
              <div>
                <div style={{ fontSize: 9.5, fontWeight: 600, letterSpacing: '0.1em', textTransform: 'uppercase', color: 'var(--txt3)', marginBottom: 4 }}>{t('result.report.assessment')}</div>
                <span className={`badge ${assessmentClass}`} style={{ fontSize: 11, padding: '4px 10px' }}>{tBadge(result.overall_assessment)}</span>
              </div>
              <div style={{ width: 1, height: 36, background: 'var(--brd)' }}/>
              <div style={{ display: 'flex', gap: 20 }}>
                {[
                  { label: t('result.report.detected'),   val: result.regions_found, color: 'var(--txt)' },
                  { label: t('result.report.suspicious'), val: result.malignant,     color: result.malignant > 0 ? 'var(--bad)' : 'var(--txt3)' },
                  { label: t('result.report.benignCount'),val: result.benign,        color: result.benign > 0   ? 'var(--warn)' : 'var(--txt3)' },
                ].map(({ label, val, color }) => (
                  <div key={label} style={{ textAlign: 'center' }}>
                    <div className="mono" style={{ fontSize: 20, fontWeight: 700, color, letterSpacing: '-0.5px', lineHeight: 1.1 }}>{val}</div>
                    <div style={{ fontSize: 10, color: 'var(--txt3)', marginTop: 2 }}>{label}</div>
                  </div>
                ))}
              </div>
              <div style={{ marginLeft: 'auto', textAlign: 'right' }}>
                <div style={{ fontSize: 9.5, fontWeight: 600, letterSpacing: '0.1em', textTransform: 'uppercase', color: 'var(--txt3)', marginBottom: 4 }}>{t('result.kpi.inference')}</div>
                <span className="mono" style={{ fontSize: 13, fontWeight: 600 }}>{fmt.time(result.inference_s)}</span>
              </div>
            </div>

            {(result.results || []).length > 0 ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10, marginBottom: 16 }}>
                {(result.results || []).map(r => {
                  const isMal = r.label === 'MALIGNANT'
                  const [x1, y1, x2, y2] = r.location.map(n => Math.round(n))
                  const barColor = isMal ? 'var(--bad)' : 'var(--warn)'
                  const bgColor  = isMal ? 'color-mix(in oklab, var(--bad) 6%, var(--ab))' : 'color-mix(in oklab, var(--warn) 6%, var(--ab))'
                  return (
                    <div key={r.region_id} style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 14, padding: '12px 16px', background: bgColor, borderRadius: 10, border: `1px solid color-mix(in oklab, ${barColor} 20%, var(--brd))` }}>
                      <div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                          <span className="mono" style={{ fontSize: 13, fontWeight: 700, color: 'var(--txt)' }}>R{r.region_id}</span>
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
                        <div style={{ height: 7, borderRadius: 999, background: 'rgba(0,0,0,0.15)', overflow: 'hidden' }}>
                          <div style={{ height: '100%', width: `${r.probability * 100}%`, background: barColor, borderRadius: 999, transition: 'width 0.4s ease' }}/>
                        </div>
                      </div>
                      <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', gap: 6 }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <span style={{ fontSize: 10, color: 'var(--txt3)', fontWeight: 600, letterSpacing: '0.08em', textTransform: 'uppercase' }}>{t('result.tip.uncertainty')}</span>
                          <span className="mono" style={{ fontSize: 12, fontWeight: 600, color: 'var(--txt2)' }}>±{(r.uncertainty * 100).toFixed(2)}%</span>
                        </div>
                        <div style={{ height: 7, borderRadius: 999, background: 'rgba(0,0,0,0.15)', overflow: 'hidden' }}>
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
            <div style={{ padding: 14, borderTop: '1px solid var(--brd)' }}>
              <pre
                className="json-block nice-scroll"
                dangerouslySetInnerHTML={{
                  __html: highlightJson({
                    overall_assessment: result.overall_assessment,
                    regions_found: result.regions_found,
                    malignant: result.malignant,
                    benign: result.benign,
                    inference_s: parseFloat(result.inference_s.toFixed(4)),
                    results: result.results.map(r => ({
                      region_id: r.region_id,
                      location: r.location.map(n => Math.round(n)),
                      label: r.label,
                      probability: parseFloat(r.probability.toFixed(4)),
                      uncertainty: parseFloat(r.uncertainty.toFixed(4)),
                    })),
                  })
                }}
              />
            </div>
          )}
        </div>
      </div>

      {toast && <Toast key={toast.k} msg={toast.msg} kind={toast.kind} onDone={() => setToast(null)}/>}
    </div>
  )
}
