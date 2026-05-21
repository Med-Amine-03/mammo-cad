'use client'

import { useState, useEffect } from 'react'
import { t, getLang, setLang, subscribeLang } from '@/lib/i18n'
import type { HealthStatus } from '@/lib/types'
import { ApiSettings } from '@/lib/storage'
import { Icon } from './Icons'

interface Props {
  open: boolean
  onClose: () => void
  onChange?: () => void
  health?: HealthStatus | null
}

export default function SettingsModal({ open, onClose, onChange, health }: Props) {
  const [, setLangState] = useState('')
  useEffect(() => subscribeLang(setLangState), [])

  const [apiBase,   setApiBase]   = useState('')
  const [forceMock, setForceMock] = useState(false)
  const [curLang,   setCurLang]   = useState('fr')

  useEffect(() => {
    if (open) {
      setApiBase(ApiSettings.getBase())
      setForceMock(ApiSettings.getForceMock())
      setCurLang(getLang())
    }
  }, [open])

  if (!open) return null

  const save = () => {
    ApiSettings.setBase(apiBase)
    ApiSettings.setForceMock(forceMock)
    setLang(curLang as import('@/lib/types').Lang)
    onChange?.()
    onClose()
  }

  const SectionLabel = ({ children }: { children: React.ReactNode }) => (
    <div style={{
      fontSize: 9.5, fontWeight: 600, letterSpacing: '0.1em',
      textTransform: 'uppercase', color: 'var(--txt3)',
      margin: '18px 0 8px',
    }}>{children}</div>
  )

  return (
    <div className="modal-bg" onClick={onClose}>
      <div className="modal" onClick={e => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose}><Icon.Close/></button>
        <h3>{t('settings.title')}</h3>
        <div className="modal-sub">{t('settings.api.help')}</div>

        {/* Language */}
        <div style={{
          fontSize: 9.5, fontWeight: 600, letterSpacing: '0.1em',
          textTransform: 'uppercase', color: 'var(--txt3)', margin: '4px 0 8px',
        }}>{t('settings.lang.label')}</div>
        <div style={{
          display: 'flex', gap: 6, padding: 3,
          background: 'var(--ab)', borderRadius: 8, border: '1px solid var(--brd)',
        }}>
          {[{ k: 'fr', l: t('settings.lang.fr') }, { k: 'en', l: t('settings.lang.en') }].map(o => (
            <button
              key={o.k}
              onClick={() => setCurLang(o.k)}
              className="mono"
              style={{
                flex: 1, padding: '7px 10px', borderRadius: 6, border: 'none', cursor: 'pointer',
                fontSize: 11.5, fontWeight: 500,
                background: curLang === o.k ? 'var(--bg2)' : 'transparent',
                color: curLang === o.k ? 'var(--txt)' : 'var(--txt3)',
                boxShadow: curLang === o.k ? '0 1px 2px rgba(0,0,0,0.2)' : 'none',
                transition: 'all 0.15s ease',
              }}
            >{o.l.toUpperCase()}</button>
          ))}
        </div>

        <SectionLabel>API</SectionLabel>
        <div style={{
          display: 'flex', alignItems: 'center', gap: 9,
          padding: '8px 12px', borderRadius: 8,
          background: 'var(--ab)', border: '1px solid var(--brd)',
          marginBottom: 12,
        }}>
          <span className={
            health?.status === 'ready' ? 'health-dot'
            : health?.status === 'loading' ? 'health-dot loading'
            : health?.status === 'offline' ? 'health-dot off'
            : 'health-dot loading'
          }/>
          <span style={{ fontSize: 11.5, color: 'var(--txt2)', fontWeight: 500 }}>
            {health?.status === 'ready' ? t('health.ready')
            : health?.status === 'loading' ? t('health.loading')
            : health?.status === 'offline' ? t('health.offline')
            : t('health.connecting')}
          </span>
          {health?.device && (
            <span style={{ marginLeft: 'auto', fontSize: 10, color: 'var(--txt3)', fontFamily: 'Geist Mono, monospace' }}>
              {health.device}
            </span>
          )}
        </div>
        <input
          className="input mono"
          value={apiBase}
          onChange={e => setApiBase(e.target.value)}
          placeholder="http://localhost:8000"
          spellCheck={false}
        />
        <div style={{ fontSize: 10.5, color: 'var(--txt3)', marginTop: 6, lineHeight: 1.5 }}>
          {t('settings.api.help')}
        </div>

        <SectionLabel>{t('settings.mock.label')}</SectionLabel>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '4px 0' }}>
          <div style={{ flex: 1, paddingRight: 12 }}>
            <div style={{ fontSize: 10.5, color: 'var(--txt3)', marginTop: 2, lineHeight: 1.5 }}>
              {t('settings.mock.help')}
            </div>
          </div>
          <button
            className={`tog${forceMock ? ' on' : ''}`}
            onClick={() => setForceMock(v => !v)}
          />
        </div>

        <div style={{ display: 'flex', gap: 8, marginTop: 18 }}>
          <button className="btn ghost" onClick={onClose} style={{ flex: 1 }}>{t('common.cancel')}</button>
          <button className="btn primary" onClick={save} style={{ flex: 1 }}>{t('settings.save')}</button>
        </div>
      </div>
    </div>
  )
}
