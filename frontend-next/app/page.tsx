'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { useRouter } from 'next/navigation'
import { t, subscribeLang } from '@/lib/i18n'
import { API } from '@/lib/api'
import { HistoryStore, LastResult } from '@/lib/storage'
import { fmt } from '@/lib/utils'
import { Icon } from '@/components/Icons'
import Toast from '@/components/Toast'
import type { ToastState } from '@/lib/types'

const VALID_EXT = ['.png', '.jpg', '.jpeg', '.dcm', '.tif', '.tiff']

function isValidFile(file: File) {
  const ext = file.name.toLowerCase().slice(file.name.lastIndexOf('.'))
  return VALID_EXT.includes(ext)
}

export default function HomePage() {
  const router = useRouter()
  const [, setLang] = useState('')
  useEffect(() => subscribeLang(setLang), [])

  const [file, setFile]               = useState<File | null>(null)
  const [previewUrl, setPreviewUrl]   = useState<string | null>(null)
  const [isDragging, setIsDragging]   = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [toast, setToast]             = useState<ToastState | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const showToast = (msg: string, kind: ToastState['kind'] = 'info') =>
    setToast({ msg, kind, k: Date.now() })

  useEffect(() => {
    if (!file) { setPreviewUrl(null); return }
    if (file.name.toLowerCase().endsWith('.dcm')) {
      setPreviewUrl('__dicom__'); return
    }
    const url = URL.createObjectURL(file)
    setPreviewUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  const onPick = (f: File | null | undefined) => {
    if (!f) return
    if (!isValidFile(f)) { showToast(t('home.toast.invalidFormat'), 'error'); return }
    setFile(f)
  }

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault(); setIsDragging(false)
    if (isProcessing) return
    onPick(e.dataTransfer.files?.[0])
  }

  const submit = useCallback(async () => {
    if (!file || isProcessing) return
    setIsProcessing(true)
    try {
      let dataUrl: string | null = null
      if (previewUrl && previewUrl !== '__dicom__') {
        dataUrl = await new Promise<string>((res, rej) => {
          const fr = new FileReader()
          fr.onload = () => res(fr.result as string)
          fr.onerror = rej
          fr.readAsDataURL(file)
        })
      }
      const result = await API.predict(file)
      if ((result as any)._dataUrl) dataUrl = (result as any)._dataUrl

      HistoryStore.add({
        id: result.id,
        filename: result.filename,
        date: Date.now(),
        overall_assessment: result.overall_assessment,
        regions_found: result.regions_found,
        malignant: result.malignant,
        benign: result.benign,
        inference_s: result.inference_s,
        _dataUrl: (dataUrl && dataUrl.length < 2_000_000) ? dataUrl : null,
        _result: result,
      })
      LastResult.save(result, dataUrl || '')

      showToast(
        `${t('home.toast.success')} · ${result.regions_found} ${t('result.kpi.regions').toLowerCase()}`,
        (result as any)._mocked ? 'warn' : 'success'
      )
      router.push('/result')
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err)
      showToast(`${t('home.toast.error')}: ${msg}`, 'error')
    } finally {
      setIsProcessing(false)
    }
  }, [file, previewUrl, isProcessing, router])

  const reset = () => { setFile(null); setPreviewUrl(null) }

  const lang = typeof window !== 'undefined' ? (localStorage.getItem('mammocad.lang') || 'fr') : 'fr'

  const pipeline = lang === 'fr' ? [
    { n:'1', title:'Prétraitement',  sub:'Redimensionnement · normalisation', c:'var(--txt2)' },
    { n:'2', title:'U-Net',          sub:'Segmentation · extraction ROI',     c:'var(--info)' },
    { n:'3', title:'EfficientNet-B3',sub:'Classification · TTA-16',           c:'var(--warn)' },
    { n:'4', title:'Rapport',          sub:'Scores · incertitude · verdict',    c:'var(--ok)' },
  ] : [
    { n:'1', title:'Preprocessing',  sub:'Resize · normalize',           c:'var(--txt2)' },
    { n:'2', title:'U-Net',          sub:'Segmentation · ROI extract',   c:'var(--info)' },
    { n:'3', title:'EfficientNet-B3',sub:'Classification · TTA-16',      c:'var(--warn)' },
    { n:'4', title:'Report',          sub:'Scores · uncertainty · verdict', c:'var(--ok)' },
  ]

  return (
    <div className="fade-in">
      <section className="hero">
        <div className="eyebrow"><span className="dot"/>{t('home.eyebrow')}</div>
        <h1 className="h1">{t('home.h1.before')}<em>{t('home.h1.em')}</em>{t('home.h1.after')}</h1>
        <p className="lead">{t('home.lead')}</p>
      </section>

      <div className="container">
        <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) 320px', gap: 20, alignItems: 'start' }}>

          {/* Drop zone card */}
          <div className="card" style={{ padding: 20 }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14 }}>
              <div className="clbl">{t('home.step1.label')}</div>
              {file && (
                <button className="btn ghost sm" onClick={reset}>
                  <Icon.Reset s={11}/> {t('home.reset')}
                </button>
              )}
            </div>

            <label
              onDrop={onDrop}
              onDragOver={e => { e.preventDefault(); setIsDragging(true) }}
              onDragLeave={() => setIsDragging(false)}
              style={{
                position: 'relative', display: 'flex', flexDirection: 'column',
                alignItems: 'center', justifyContent: 'center', gap: 14,
                width: '100%', minHeight: 380, padding: 28,
                border: `2px dashed ${isDragging ? 'var(--info)' : 'var(--brd2)'}`,
                borderRadius: 14,
                background: isDragging ? 'var(--info-bg)' : 'var(--ab)',
                cursor: isProcessing ? 'wait' : 'pointer',
                transition: 'all 0.18s', overflow: 'hidden',
              }}
            >
              <input
                ref={inputRef}
                type="file"
                accept=".png,.jpg,.jpeg,.dcm,.tif,.tiff"
                onChange={e => { onPick(e.target.files?.[0]); e.target.value = '' }}
                disabled={isProcessing}
                style={{ position: 'absolute', inset: 0, opacity: 0, cursor: 'pointer' }}
              />

              {!file && (
                <>
                  <div style={{
                    width: 56, height: 56, borderRadius: 14,
                    background: 'var(--ab2)', border: '1px solid var(--brd)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--txt2)',
                  }}>
                    <Icon.Upload s={22}/>
                  </div>
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--txt)', marginBottom: 4 }}>
                      {isDragging ? t('home.dropzone.dragging') : t('home.dropzone.cta')}
                    </div>
                    <div style={{ fontSize: 12, color: 'var(--txt3)' }}>
                      {t('home.dropzone.or')} · {t('home.dropzone.formats')}
                    </div>
                  </div>
                  <div style={{ display: 'flex', gap: 6, marginTop: 4 }}>
                    {['DICOM','PNG','JPG','TIFF','INbreast'].map(tag => (
                      <span key={tag} style={{
                        fontSize: 9.5, fontFamily: 'Geist Mono, monospace',
                        padding: '3px 8px', borderRadius: 999,
                        background: 'var(--ab2)', border: '1px solid var(--brd)',
                        color: 'var(--txt3)', letterSpacing: '0.04em',
                      }}>{tag}</span>
                    ))}
                  </div>
                </>
              )}

              {file && previewUrl === '__dicom__' && (
                <div style={{ textAlign: 'center', padding: '40px 20px' }}>
                  <div style={{
                    width: 56, height: 56, borderRadius: 14, margin: '0 auto 14px',
                    background: 'var(--info-bg)',
                    border: '1px solid color-mix(in oklab, var(--info) 28%, transparent)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--info)',
                  }}><Icon.Check s={22}/></div>
                  <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--txt)', marginBottom: 4 }}>DICOM ✓</div>
                  <div style={{ fontSize: 11.5, color: 'var(--txt3)', fontFamily: 'Geist Mono, monospace' }}>{file.name}</div>
                  <div style={{ fontSize: 10.5, color: 'var(--txt3)', marginTop: 4 }}>{t('home.dicom.hint')}</div>
                </div>
              )}

              {file && previewUrl && previewUrl !== '__dicom__' && (
                <div style={{ position: 'relative', width: '100%', maxHeight: 360 }}>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={previewUrl}
                    alt="Preview"
                    style={{ display: 'block', maxWidth: '100%', maxHeight: 360, margin: '0 auto', borderRadius: 8, objectFit: 'contain' }}
                  />
                  {isProcessing && (
                    <div style={{
                      position: 'absolute', inset: 0,
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      background: 'rgba(0,0,0,0.55)', backdropFilter: 'blur(2px)', borderRadius: 8,
                    }}>
                      <div style={{ textAlign: 'center', color: '#fff' }}>
                        <div style={{
                          width: 32, height: 32, margin: '0 auto 12px',
                          border: '2.5px solid rgba(255,255,255,0.2)',
                          borderTopColor: '#fff', borderRadius: '50%',
                          animation: 'spin 0.8s linear infinite',
                        }}/>
                        <div style={{ fontSize: 12, fontWeight: 500 }}>{t('home.running')}</div>
                        <div style={{ fontSize: 10.5, opacity: 0.7, marginTop: 4 }}>U-Net → EfficientNet-B3</div>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </label>

            <div style={{ display: 'flex', gap: 10, marginTop: 16 }}>
              <button
                className="btn primary lg"
                onClick={submit}
                disabled={!file || isProcessing}
                style={{ flex: 1 }}
              >
                {isProcessing ? (
                  <><span style={{
                    display: 'inline-block', width: 13, height: 13,
                    border: '2px solid currentColor', borderTopColor: 'transparent',
                    borderRadius: '50%', animation: 'spin 0.8s linear infinite',
                  }}/> {t('home.running')}</>
                ) : (
                  <><Icon.Play s={11}/> POST /predict</>
                )}
              </button>
              <button className="btn ghost lg" onClick={reset} disabled={!file || isProcessing}>
                {t('home.reset')}
              </button>
            </div>

            {file && !isProcessing && (
              <div style={{
                marginTop: 10, padding: '8px 12px',
                background: 'var(--ab)', border: '1px solid var(--brd)', borderRadius: 7,
                fontFamily: 'Geist Mono, monospace', fontSize: 10.5, color: 'var(--txt3)',
                display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap',
              }}>
                <span>📎 {file.name}</span>
                <span>·</span>
                <span>{fmt.bytes(file.size)}</span>
                <span>·</span>
                <span>→ <span style={{ color: 'var(--info)' }}>{API.base()}/predict</span></span>
              </div>
            )}
          </div>

          {/* Side info */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <div className="card hoverable">
              <div className="clbl" style={{ marginBottom: 10 }}>Pipeline</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {pipeline.map(p => (
                  <div key={p.n} style={{ display: 'flex', alignItems: 'center', gap: 11 }}>
                    <div style={{
                      width: 24, height: 24, borderRadius: 6,
                      background: `color-mix(in oklab, ${p.c} 14%, transparent)`,
                      color: p.c,
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      fontSize: 11, fontWeight: 700, fontFamily: 'Geist Mono, monospace',
                      flexShrink: 0,
                    }}>{p.n}</div>
                    <div>
                      <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--txt)', lineHeight: 1.2 }}>{p.title}</div>
                      <div style={{ fontSize: 10.5, color: 'var(--txt3)', marginTop: 1 }}>{p.sub}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

          </div>
        </div>
      </div>

      {toast && (
        <Toast key={toast.k} msg={toast.msg} kind={toast.kind} onDone={() => setToast(null)}/>
      )}
    </div>
  )
}
