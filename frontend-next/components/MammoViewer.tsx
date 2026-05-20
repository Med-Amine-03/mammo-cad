'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import type { Region } from '@/lib/types'
import { Icon } from './Icons'
import { t, tBadge } from '@/lib/i18n'

interface Props {
  imageSrc: string
  regions?: Region[]
  imageSize?: [number, number] | null
  hoveredId?: number | null
  onHoverChange?: (id: number | null) => void
  showBoxes?: boolean
  showLabels?: boolean
}

interface Transform { scale: number; x: number; y: number }
interface Tooltip { x: number; y: number; region: Region }

export default function MammoViewer({
  imageSrc, regions = [], imageSize = null,
  hoveredId, onHoverChange, showBoxes = true, showLabels = true,
}: Props) {
  const wrapRef  = useRef<HTMLDivElement>(null)
  const stageRef = useRef<HTMLDivElement>(null)
  const imgRef   = useRef<HTMLImageElement>(null)

  const [natural, setNatural] = useState<{ w: number; h: number } | null>(
    imageSize ? { w: imageSize[0], h: imageSize[1] } : null
  )
  const [tooltip, setTooltip]     = useState<Tooltip | null>(null)
  const [transform, setTransform] = useState<Transform>({ scale: 1, x: 0, y: 0 })
  const [isPanning, setIsPanning] = useState(false)
  const panStartRef = useRef({ x: 0, y: 0, tx: 0, ty: 0 })

  useEffect(() => { setTransform({ scale: 1, x: 0, y: 0 }); setTooltip(null) }, [imageSrc])

  const onLoad = () => {
    const img = imgRef.current
    if (img) setNatural({ w: img.naturalWidth, h: img.naturalHeight })
  }

  const zoomIn  = () => setTransform(t => ({ ...t, scale: Math.min(t.scale * 1.3, 8) }))
  const zoomOut = () => setTransform(t => ({ ...t, scale: Math.max(t.scale / 1.3, 0.25) }))
  const reset   = () => setTransform({ scale: 1, x: 0, y: 0 })

  // Attach a non-passive native wheel listener so we can call preventDefault()
  // only when Ctrl is held — normal scroll passes through to the page.
  useEffect(() => {
    const stage = stageRef.current
    if (!stage) return
    const handler = (e: WheelEvent) => {
      if (!e.ctrlKey) return          // let the page scroll normally
      e.preventDefault()              // consume the event for zoom
      const rect = stage.getBoundingClientRect()
      const cx = e.clientX - rect.left - rect.width / 2
      const cy = e.clientY - rect.top - rect.height / 2
      const factor = e.deltaY > 0 ? 0.9 : 1.1
      setTransform(t => {
        const newScale = Math.min(Math.max(t.scale * factor, 0.25), 8)
        const diff = newScale / t.scale
        return { scale: newScale, x: cx - (cx - t.x) * diff, y: cy - (cy - t.y) * diff }
      })
    }
    stage.addEventListener('wheel', handler, { passive: false })
    return () => stage.removeEventListener('wheel', handler)
  }, [])

  // Keep onWheel as a no-op stub so the JSX prop doesn't need to change
  const onWheel = useCallback((_e: React.WheelEvent) => {}, [])

  const onMouseDown = (e: React.MouseEvent) => {
    if (e.button !== 0) return
    e.preventDefault()
    setIsPanning(true)
    panStartRef.current = { x: e.clientX, y: e.clientY, tx: transform.x, ty: transform.y }
  }
  const onMouseMove = (e: React.MouseEvent) => {
    if (!isPanning) return
    const { x, y, tx, ty } = panStartRef.current
    setTransform(t => ({ ...t, x: tx + (e.clientX - x), y: ty + (e.clientY - y) }))
  }
  const stopPan = () => setIsPanning(false)

  const positionTooltip = (e: React.MouseEvent): { x: number; y: number } | null => {
    const wrap = wrapRef.current
    if (!wrap) return null
    const r = wrap.getBoundingClientRect()
    let x = e.clientX - r.left + 15
    let y = e.clientY - r.top - 15
    if (x + 240 > r.width) x = e.clientX - r.left - 250
    if (y < 10) y = e.clientY - r.top + 20
    if (y + 280 > r.height) y = r.height - 290
    return { x, y }
  }

  const onBoxEnter = (region: Region, e: React.MouseEvent) => {
    e.stopPropagation()
    onHoverChange?.(region.region_id)
    const pos = positionTooltip(e)
    if (pos) setTooltip({ ...pos, region })
  }
  const onBoxMove = (region: Region, e: React.MouseEvent) => {
    e.stopPropagation()
    const pos = positionTooltip(e)
    if (pos) setTooltip({ ...pos, region })
  }
  const onBoxLeave = (e: React.MouseEvent) => {
    e.stopPropagation()
    onHoverChange?.(null)
    setTooltip(null)
  }

  if (!imageSrc) return null

  const W = natural?.w || 1, H = natural?.h || 1

  const geom = (loc: number[]) => {
    const [x1, y1, x2, y2] = loc
    const x = Math.min(x1, x2), y = Math.min(y1, y2)
    return { x, y, w: Math.abs(x2 - x1), h: Math.abs(y2 - y1) }
  }

  const stats = (region: Region) => {
    const { w, h } = geom(region.location)
    const area = w * h
    const coverage = W && H ? (area / (W * H)) * 100 : 0
    return { w, h, area, coverage }
  }

  const cursor = isPanning ? 'grabbing' : transform.scale > 1 ? 'grab' : 'default'

  return (
    <div ref={wrapRef} style={{
      position: 'relative', width: '100%', height: 580,
      background: '#000', borderRadius: 12, overflow: 'hidden',
      border: '1px solid var(--brd)', userSelect: 'none',
    }}>
      {/* Zoom controls */}
      <div style={{
        position: 'absolute', top: 12, right: 12, zIndex: 30,
        display: 'flex', alignItems: 'center', gap: 2,
        background: 'rgba(14,16,19,0.92)', border: '1px solid var(--brd)',
        borderRadius: 8, padding: 3, backdropFilter: 'blur(8px)',
      }}>
        <button className="icon-btn" onClick={zoomOut} style={{ width: 28, height: 28 }}><Icon.ZoomOut s={14}/></button>
        <span className="mono" style={{ fontSize: 10, color: 'var(--txt3)', width: 42, textAlign: 'center' }}>
          {Math.round(transform.scale * 100)}%
        </span>
        <button className="icon-btn" onClick={zoomIn} style={{ width: 28, height: 28 }}><Icon.ZoomIn s={14}/></button>
        <span style={{ width: 1, height: 14, background: 'var(--brd)', margin: '0 4px' }}/>
        <button className="icon-btn" onClick={reset} style={{ width: 28, height: 28 }}><Icon.Reset s={13}/></button>
      </div>

      {transform.scale > 1 && (
        <div style={{
          position: 'absolute', top: 12, left: 12, zIndex: 30,
          display: 'flex', alignItems: 'center', gap: 6,
          background: 'rgba(14,16,19,0.92)', border: '1px solid var(--brd)',
          borderRadius: 8, padding: '4px 9px', backdropFilter: 'blur(8px)',
          fontSize: 9.5, color: 'var(--txt3)',
        }}>
          <Icon.Move s={11}/> Drag to pan
        </div>
      )}

      {/* Stage */}
      <div ref={stageRef}
        onWheel={onWheel}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={stopPan}
        onMouseLeave={stopPan}
        style={{
          position: 'absolute', inset: 0,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          padding: 16, cursor, overflow: 'hidden',
        }}
      >
        <div style={{
          transform: `translate(${transform.x}px, ${transform.y}px) scale(${transform.scale})`,
          transformOrigin: 'center center',
          transition: isPanning ? 'none' : 'transform 0.15s ease-out',
          position: 'relative', maxWidth: '100%', maxHeight: '100%',
        }}>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img ref={imgRef} src={imageSrc} alt="Mammogram" onLoad={onLoad}
            draggable={false} crossOrigin="anonymous"
            style={{
              display: 'block', maxWidth: '100%', maxHeight: 548,
              objectFit: 'contain', borderRadius: 8, pointerEvents: 'none',
              boxShadow: '0 12px 32px rgba(0,0,0,0.55)',
            }}
          />

          {showBoxes && natural && (
            <div style={{ position: 'absolute', inset: 0 }}>
              {regions.map(r => {
                const { x, y, w, h } = geom(r.location)
                const isMal = r.label === 'MALIGNANT'
                const color = isMal ? '#ef4444' : '#fbbf24'
                const isHov = hoveredId === r.region_id
                return (
                  <div key={r.region_id}
                    style={{
                      position: 'absolute',
                      left: `${(x / W) * 100}%`, top: `${(y / H) * 100}%`,
                      width: `${(w / W) * 100}%`, height: `${(h / H) * 100}%`,
                      border: 'none',
                      background: 'transparent',
                      borderRadius: 3, cursor: 'crosshair',
                      pointerEvents: isPanning ? 'none' : 'auto',
                    }}
                    onMouseEnter={e => onBoxEnter(r, e)}
                    onMouseMove={e => onBoxMove(r, e)}
                    onMouseLeave={onBoxLeave}
                    onMouseDown={e => e.stopPropagation()}
                  />
                )
              })}
            </div>
          )}
        </div>
      </div>

      {/* Footer hint */}
      <div style={{
        position: 'absolute', bottom: 10, right: 12, zIndex: 20,
        fontFamily: 'Geist Mono, monospace', fontSize: 9,
        color: 'rgba(163,165,171,0.55)', background: 'rgba(14,16,19,0.8)',
        border: '1px solid var(--brd)', borderRadius: 6, padding: '3px 8px',
        backdropFilter: 'blur(6px)',
      }}>
        Ctrl + scroll to zoom · drag to pan · hover boxes for details
      </div>

      {/* Tooltip */}
      {tooltip && (() => {
        const r = tooltip.region
        const isMal = r.label === 'MALIGNANT'
        const s = stats(r)
        return (
          <div style={{
            position: 'absolute', left: tooltip.x, top: tooltip.y, zIndex: 50,
            background: 'rgba(14,16,19,0.96)', border: '1px solid rgba(255,255,255,0.10)',
            color: '#e8e9ec', borderRadius: 12, padding: 14,
            minWidth: 220, maxWidth: 240, backdropFilter: 'blur(10px)',
            boxShadow: '0 18px 40px rgba(0,0,0,0.55)', pointerEvents: 'none',
            animation: 'fadeIn 0.15s ease',
          }}>
            <div style={{
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              paddingBottom: 8, marginBottom: 10, borderBottom: '1px solid rgba(255,255,255,0.06)',
            }}>
              <span style={{ fontSize: 9.5, fontWeight: 600, letterSpacing: '0.11em', textTransform: 'uppercase', color: '#6b6d75' }}>
                Cluster #{r.region_id}
              </span>
              <span className={`badge ${r.label.toLowerCase()}`} style={{ fontSize: 9.5, padding: '2px 7px' }}>
                {tBadge(r.label)}
              </span>
            </div>
            <div style={{ marginBottom: 10 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                <span style={{ fontSize: 10, color: '#6b6d75' }}>{t('result.tip.probability')}</span>
                <span className="mono" style={{ fontSize: 11.5, fontWeight: 600, color: '#e8e9ec' }}>
                  {(r.probability * 100).toFixed(1)}%
                </span>
              </div>
              <div style={{ width: '100%', height: 6, borderRadius: 999, background: 'rgba(255,255,255,0.06)', overflow: 'hidden' }}>
                <div style={{
                  height: '100%', width: `${r.probability * 100}%`,
                  background: isMal
                    ? 'linear-gradient(90deg, rgba(239,68,68,0.7), #ef4444)'
                    : 'linear-gradient(90deg, rgba(251,191,36,0.7), #fbbf24)',
                  borderRadius: 999,
                }}/>
              </div>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
              <span style={{ fontSize: 10, color: '#6b6d75' }}>{t('result.tip.uncertainty')}</span>
              <span className="mono" style={{ fontSize: 11, color: '#fbbf24' }}>±{(r.uncertainty * 100).toFixed(2)}%</span>
            </div>
            <div style={{ paddingTop: 9, marginTop: 2, borderTop: '1px solid rgba(255,255,255,0.06)' }}>
              <div style={{ fontSize: 9, fontWeight: 600, letterSpacing: '0.1em', textTransform: 'uppercase', color: '#6b6d75', marginBottom: 6 }}>Region size</div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, marginBottom: 6 }}>
                {[['Width', s.w], ['Height', s.h]].map(([label, val]) => (
                  <div key={label as string} style={{ background: 'rgba(255,255,255,0.025)', borderRadius: 5, padding: '5px 8px' }}>
                    <div style={{ fontSize: 9, color: '#6b6d75' }}>{label}</div>
                    <div className="mono" style={{ fontSize: 10.5, color: '#e8e9ec' }}>{Math.round(val as number)}px</div>
                  </div>
                ))}
              </div>
              <div style={{ background: 'rgba(255,255,255,0.025)', borderRadius: 5, padding: '5px 8px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 1 }}>
                  <span style={{ fontSize: 9, color: '#6b6d75' }}>Area</span>
                  <span className="mono" style={{ fontSize: 10.5, color: '#e8e9ec' }}>{Math.round(s.area).toLocaleString()}px²</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ fontSize: 9, color: '#6b6d75' }}>Coverage</span>
                  <span className="mono" style={{ fontSize: 10.5, color: 'var(--acc)' }}>{s.coverage.toFixed(3)}%</span>
                </div>
              </div>
            </div>
            <div style={{ paddingTop: 8, marginTop: 8, borderTop: '1px solid rgba(255,255,255,0.06)' }}>
              <div style={{ fontSize: 9, fontWeight: 600, letterSpacing: '0.1em', textTransform: 'uppercase', color: '#6b6d75', marginBottom: 3 }}>
                {t('result.tip.bbox')}
              </div>
              <div className="mono" style={{ fontSize: 10, color: 'rgba(163,165,171,0.85)' }}>
                [{r.location.map(v => Math.round(v)).join(', ')}]
              </div>
            </div>
          </div>
        )
      })()}
    </div>
  )
}
