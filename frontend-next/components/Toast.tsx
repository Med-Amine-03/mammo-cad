'use client'

import { useEffect } from 'react'
import { Icon } from './Icons'

interface Props {
  msg: string
  kind?: 'info' | 'success' | 'warn' | 'error'
  onDone?: () => void
}

export default function Toast({ msg, kind = 'info', onDone }: Props) {
  useEffect(() => {
    if (!msg) return
    const id = setTimeout(() => onDone?.(), 3200)
    return () => clearTimeout(id)
  }, [msg, onDone])

  if (!msg) return null

  const colors: Record<string, { bg: string; fg: string; ic: React.ReactNode }> = {
    info:    { bg: 'var(--info-bg)',  fg: 'var(--info)',  ic: <Icon.Check/> },
    success: { bg: 'var(--ok-bg)',    fg: 'var(--ok)',    ic: <Icon.Check/> },
    warn:    { bg: 'var(--warn-bg)',  fg: 'var(--warn)',  ic: <Icon.Warn/> },
    error:   { bg: 'var(--bad-bg)',   fg: 'var(--bad)',   ic: <Icon.Warn/> },
  }
  const c = colors[kind] || colors.info

  return (
    <div style={{
      position: 'fixed', bottom: 24, left: '50%', transform: 'translateX(-50%)',
      zIndex: 300,
      padding: '10px 16px',
      borderRadius: 10,
      background: c.bg,
      color: c.fg,
      border: `1px solid color-mix(in oklab, ${c.fg} 30%, transparent)`,
      backdropFilter: 'blur(10px)',
      fontSize: 12.5, fontWeight: 500,
      display: 'flex', alignItems: 'center', gap: 8,
      boxShadow: '0 12px 32px rgba(0,0,0,0.3)',
      animation: 'fadeIn 0.2s ease',
      whiteSpace: 'nowrap',
    }}>
      {c.ic} {msg}
    </div>
  )
}
