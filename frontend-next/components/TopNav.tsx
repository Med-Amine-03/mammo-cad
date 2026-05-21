'use client'

import { t, subscribeLang } from '@/lib/i18n'
import { Icon } from './Icons'
import { useState, useEffect } from 'react'

interface Props {
  page: string
  onNavigate: (path: string) => void
  onOpenSettings: () => void
  onToggleTheme: () => void
  theme: 'dark' | 'light'
}

export default function TopNav({ page, onNavigate, onOpenSettings, onToggleTheme, theme }: Props) {
  const [, setLang] = useState('')
  useEffect(() => subscribeLang(setLang), [])

  const links = [
    { id: 'home',    path: '/',        label: t('nav.home') },
    { id: 'result',  path: '/result',  label: t('nav.result') },
    { id: 'history', path: '/history', label: t('nav.history') },
    { id: 'models',  path: '/models',  label: t('nav.models') },
  ]

  return (
    <nav className="nav">
      {/* Brand / Logo */}
      <div
        className="brand brand-logo"
        onClick={() => onNavigate('/')}
        role="button"
        tabIndex={0}
        title="MammoCAD"
        onKeyDown={e => e.key === 'Enter' && onNavigate('/')}
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={theme === 'light' ? '/logo-wordmark-light.png' : '/logo-wordmark-dark.png'}
          alt="MammoCAD"
          style={{ height: 22, display: 'block' }}
          onError={e => {
            const img = e.target as HTMLImageElement
            img.style.display = 'none'
            const next = img.nextElementSibling as HTMLElement | null
            if (next) next.style.display = 'block'
          }}
          onLoad={e => {
            const img = e.target as HTMLImageElement
            img.style.display = 'block'
            const next = img.nextElementSibling as HTMLElement | null
            if (next) next.style.display = 'none'
          }}
        />
        {/* Fallback text shown only if logo fails to load */}
        <span style={{ fontSize: 15, fontWeight: 700, letterSpacing: '-0.3px', display: 'none' }}>MammoCAD</span>
      </div>

      {/* Nav links */}
      <div className="nav-links">
        {links.map(l => (
          <span
            key={l.id}
            className={`nl${page === l.id ? ' on' : ''}`}
            onClick={() => onNavigate(l.path)}
            role="button"
            tabIndex={0}
            onKeyDown={e => e.key === 'Enter' && onNavigate(l.path)}
          >
            {l.label}
          </span>
        ))}
      </div>

      {/* Right tools */}
      <div className="nav-tools">
        <button className="icon-btn" onClick={onOpenSettings} title={t('settings.title')}>
          <svg width={15} height={15} viewBox="0 0 15 15" fill="none">
            <path d="M5.5 2v11M9.5 2v11" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round"/>
            <rect x="3.8" y="4.2" width="3.4" height="2.6" rx="1" fill="var(--bg2)" stroke="currentColor" strokeWidth="1.1"/>
            <rect x="7.8" y="8.2" width="3.4" height="2.6" rx="1" fill="var(--bg2)" stroke="currentColor" strokeWidth="1.1"/>
          </svg>
        </button>
        <button className="icon-btn" onClick={onToggleTheme} title={t('settings.theme.label')}>
          {theme === 'light' ? <Icon.Moon s={14}/> : <Icon.Sun s={14}/>}
        </button>
      </div>
    </nav>
  )
}
