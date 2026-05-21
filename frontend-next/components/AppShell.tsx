'use client'

import { useState, useEffect, useCallback } from 'react'
import { usePathname, useRouter } from 'next/navigation'
import type { HealthStatus, ToastState } from '@/lib/types'
import { ThemeStore } from '@/lib/storage'
import { ApiSettings } from '@/lib/storage'
import { API } from '@/lib/api'
import TopNav from './TopNav'
import FooterBar from './FooterBar'
import Toast from './Toast'
import SettingsModal from './SettingsModal'

export default function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname()
  const router = useRouter()

  const [theme, setTheme]             = useState<'dark' | 'light'>('dark')
  const [health, setHealth]           = useState<HealthStatus | null>(null)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [toast, setToast]             = useState<ToastState | null>(null)

  // Init theme on mount
  useEffect(() => {
    ThemeStore.init()
    setTheme(ThemeStore.get())
  }, [])

  // Poll /health
  useEffect(() => {
    let mounted = true
    let ctrl: AbortController | undefined
    const check = async () => {
      ctrl?.abort()
      ctrl = new AbortController()
      const h = await API.health(ctrl.signal)
      if (mounted) setHealth(h)
    }
    check()
    const id = setInterval(check, 12000)
    return () => { mounted = false; clearInterval(id); ctrl?.abort() }
  }, [])

  const showToast = useCallback((msg: string, kind: ToastState['kind'] = 'info') => {
    setToast({ msg, kind, k: Date.now() })
  }, [])

  const onToggleTheme = () => {
    ThemeStore.toggle()
    setTheme(ThemeStore.get())
  }

  const onSettingsChange = () => {
    showToast('Settings saved · reconnecting…', 'success')
    API.health().then(setHealth)
  }

  const navigate = (path: string) => {
    window.scrollTo({ top: 0, behavior: 'smooth' })
    router.push(path)
  }

  // Derive active page from pathname
  const page = pathname === '/' ? 'home'
    : pathname.startsWith('/result') ? 'result'
    : pathname.startsWith('/history') ? 'history'
    : pathname.startsWith('/models') ? 'models'
    : 'home'

  return (
    <div className="app">
      <div className="grid-bg"/>
      <div className="grid-fade"/>

      <div className="layer" style={{ display: 'flex', flexDirection: 'column', flex: 1 }}>
        <TopNav
          page={page}
          onNavigate={navigate}
          onOpenSettings={() => setSettingsOpen(true)}
          onToggleTheme={onToggleTheme}
          theme={theme}
        />

        <main style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
          {children}
        </main>

        <FooterBar/>
      </div>

      <SettingsModal
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        onChange={onSettingsChange}
        health={health}
      />

      {toast && (
        <Toast
          key={toast.k}
          msg={toast.msg}
          kind={toast.kind}
          onDone={() => setToast(null)}
        />
      )}
    </div>
  )
}

// Export showToast so pages can call it — but since we can't pass it down easily,
// pages get their own local toast via a shared context approach.
// For simplicity, pages import and use a ToastContext or manage their own local toasts.
export type { ToastState }
