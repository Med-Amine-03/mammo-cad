'use client'

import { useState, useEffect, useMemo, useCallback } from 'react'
import { useRouter } from 'next/navigation'
import { t, tBadge, subscribeLang } from '@/lib/i18n'
import { API } from '@/lib/api'
import { fmt } from '@/lib/utils'
import type { HistoryItem, ToastState } from '@/lib/types'
import { Icon } from '@/components/Icons'
import Toast from '@/components/Toast'

function StatBox({ value, label, color }: { value: number; label: string; color: string }) {
  return (
    <div className="card hoverable" style={{ padding: '14px 16px' }}>
      <div className="mono" style={{ fontSize: 24, fontWeight: 700, letterSpacing: '-0.5px', color, lineHeight: 1.1 }}>{value}</div>
      <div className="clbl" style={{ marginTop: 4 }}>{label}</div>
    </div>
  )
}

function Spinner() {
  return (
    <div style={{ width: 28, height: 28, margin: '0 auto', border: '2.5px solid var(--brd2)', borderTopColor: 'var(--acc)', borderRadius: '50%', animation: 'spin 0.8s linear infinite' }}/>
  )
}

export default function HistoryPage() {
  const router = useRouter()
  const [, setLang] = useState('')
  useEffect(() => subscribeLang(setLang), [])

  const [items, setItems]           = useState<HistoryItem[]>([])
  const [source, setSource]         = useState<string | null>(null)
  const [listLoading, setListLoading] = useState(true)
  const [listError, setListError]   = useState<string | null>(null)
  const [filter, setFilter]         = useState('all')
  const [query, setQuery]           = useState('')
  const [toast, setToast]           = useState<ToastState | null>(null)

  const showToast = (msg: string, kind: ToastState['kind'] = 'info') =>
    setToast({ msg, kind, k: Date.now() })

  const fetchList = useCallback(async () => {
    setListLoading(true); setListError(null)
    try {
      const { items: it, source: src, error } = await API.history()
      setItems(it || []); setSource(src)
      if (error && (!it || it.length === 0)) setListError(error)
    } catch (e: unknown) {
      setListError(e instanceof Error ? e.message : String(e))
      setItems([])
    } finally { setListLoading(false) }
  }, [])

  useEffect(() => { fetchList() }, [fetchList])

  const filtered = useMemo(() => items.filter(it => {
    if (filter !== 'all' && (it.prediction || '').toUpperCase() !== filter) return false
    if (query) {
      const q = query.toLowerCase()
      if (!(it.filename?.toLowerCase().includes(q) || it.job_id?.toLowerCase().includes(q))) return false
    }
    return true
  }), [items, filter, query])

  const stats = useMemo(() => ({
    total: items.length,
    mal: items.filter(i => i.prediction === 'MALIGNANT').length,
    ben: items.filter(i => i.prediction === 'BENIGN').length,
  }), [items])

  const filters = [
    { id: 'all',       label: t('history.filter.all') },
    { id: 'MALIGNANT', label: t('history.filter.malignant') },
    { id: 'BENIGN',    label: t('history.filter.benign') },
  ]

  return (
    <div className="fade-in">
      <section className="hero">
        <div className="eyebrow">
          <span className="dot"/>
          {t('history.eyebrow')}{source && <span style={{ marginLeft: 8, opacity: 0.7 }}>· {source}</span>}
        </div>
        <h1 className="h1" style={{ fontSize: 30 }}>{t('history.h1')}</h1>
        <p className="lead" style={{ maxWidth: 520 }}>{t('history.lead')}</p>
      </section>

      <div className="container">
        {/* Stats */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12, marginBottom: 16 }}>
          <StatBox value={stats.total} label={t('history.stats.total')}     color="var(--txt)"/>
          <StatBox value={stats.mal}   label={t('history.stats.malignant')} color="var(--bad)"/>
          <StatBox value={stats.ben}   label={t('history.stats.benign')}    color="var(--warn)"/>
        </div>

        {/* Filter row */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 14, flexWrap: 'wrap' }}>
          {filters.map(f => (
            <button key={f.id} className={`btn sm${filter === f.id ? ' primary' : ' ghost'}`} onClick={() => setFilter(f.id)}>
              {f.label}
            </button>
          ))}
          <div style={{ flex: 1, minWidth: 200, marginLeft: 8, display: 'flex', alignItems: 'center', gap: 8, background: 'var(--ab)', border: '1px solid var(--brd2)', borderRadius: 8, padding: '0 12px', height: 32 }}>
            <Icon.Search s={12}/>
            <input
              type="text"
              placeholder={t('history.search')}
              value={query}
              onChange={e => setQuery(e.target.value)}
              style={{ flex: 1, background: 'transparent', border: 'none', outline: 'none', color: 'var(--txt)', fontSize: 12, fontFamily: 'inherit' }}
            />
          </div>
          <button className="btn ghost sm" onClick={fetchList} disabled={listLoading}>
            <Icon.Reset s={11}/> {listLoading ? t('history.refreshing') : t('history.refresh')}
          </button>
        </div>

        {/* Offline banner */}
        {source === 'offline' && (
          <div className="disc" style={{ marginBottom: 12 }}>
            <div className="disc-icon"><Icon.Warn s={13}/></div>
            <div><strong>{t('history.offline.title')}</strong> {t('history.offline.desc')}</div>
          </div>
        )}

        {/* Content */}
        {listLoading && items.length === 0 ? (
          <div className="card" style={{ padding: '56px 24px', textAlign: 'center' }}>
            <Spinner/>
            <div style={{ fontSize: 12, color: 'var(--txt3)', marginTop: 14 }}>Fetching /history…</div>
          </div>
        ) : listError && items.length === 0 ? (
          <div className="card" style={{ padding: '40px 24px', textAlign: 'center' }}>
            <div style={{ width: 48, height: 48, margin: '0 auto 14px', borderRadius: 12, background: 'var(--bad-bg)', color: 'var(--bad)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}><Icon.Warn s={20}/></div>
            <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 4 }}>{t('history.error.title')}</div>
            <div style={{ fontSize: 12, color: 'var(--txt3)', marginBottom: 18, fontFamily: 'Geist Mono, monospace' }}>{listError}</div>
            <button className="btn primary" onClick={fetchList}><Icon.Reset s={12}/> {t('history.error.retry')}</button>
          </div>
        ) : filtered.length === 0 ? (
          <div className="card" style={{ padding: '56px 24px', textAlign: 'center' }}>
            <div style={{ width: 48, height: 48, margin: '0 auto 14px', borderRadius: 12, background: 'var(--ab)', border: '1px solid var(--brd)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--txt3)' }}><Icon.Json s={20}/></div>
            <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 4 }}>
              {items.length === 0 ? t('history.empty.none.title') : t('history.empty.match.title')}
            </div>
            <div style={{ fontSize: 12, color: 'var(--txt3)', marginBottom: 18 }}>
              {items.length === 0 ? t('history.empty.none.desc') : t('history.empty.match.desc')}
            </div>
            {items.length === 0 && (
              <button className="btn primary" onClick={() => router.push('/')}>
                <Icon.Upload s={12}/> {t('history.empty.none.cta')}
              </button>
            )}
          </div>
        ) : (
          <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
            <table className="table">
              <thead>
                <tr>
                  <th style={{ width: 100 }}>{t('history.col.id')}</th>
                  <th>{t('history.col.file')}</th>
                  <th style={{ width: 140 }}>{t('history.col.date')}</th>
                  <th style={{ width: 130 }}>{t('history.col.prediction')}</th>
                  <th style={{ width: 80 }}>{t('history.col.regions')}</th>
                  <th style={{ width: 110 }}>{t('history.col.confidence')}</th>
                  <th style={{ width: 90 }}>{t('history.col.time')}</th>
                  <th style={{ width: 36 }}/>
                </tr>
              </thead>
              <tbody>
                {filtered.map(it => {
                  const cls = it.prediction === 'MALIGNANT' ? 'suspicious'
                            : it.prediction === 'BENIGN'    ? 'benign' : 'normal'
                  return (
                    <tr key={it.job_id} onClick={() => router.push(`/history/${it.job_id}`)}>
                      <td className="mono" style={{ color: 'var(--txt3)', fontSize: 10.5 }}>{fmt.shortId(it.job_id)}</td>
                      <td style={{ maxWidth: 240, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', fontWeight: 500 }}>{it.filename}</td>
                      <td className="mono" style={{ color: 'var(--txt3)', fontSize: 10.5 }}>{fmt.date(it.timestamp)}</td>
                      <td><span className={`badge ${cls}`}>{tBadge(it.prediction || 'UNKNOWN')}</span></td>
                      <td className="mono" style={{ fontWeight: 600 }}>{it.n_regions}</td>
                      <td className="mono" style={{ color: 'var(--txt2)' }}>
                        {typeof it.confidence === 'number' ? (it.confidence * 100).toFixed(1) + '%' : '—'}
                      </td>
                      <td className="mono" style={{ color: 'var(--txt2)', fontSize: 11 }}>{fmt.time(it.processing_time || 0)}</td>
                      <td style={{ textAlign: 'right', color: 'var(--txt3)' }}><Icon.ArrowRight s={12}/></td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {toast && <Toast key={toast.k} msg={toast.msg} kind={toast.kind} onDone={() => setToast(null)}/>}
    </div>
  )
}
