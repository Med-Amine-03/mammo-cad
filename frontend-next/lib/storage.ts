import type { PredictResult } from './types'

const LS_HISTORY    = 'mammocad.history'
const LS_THEME      = 'mammocad.theme'
const LS_LAST_RESULT= 'mammocad.lastResult'
const LS_API_BASE   = 'mammocad.apiBase'
const LS_USE_MOCK   = 'mammocad.forceMock'

// ── History ──────────────────────────────────────────────────────────────
export interface LocalHistoryEntry {
  id: string
  filename: string
  date: number
  overall_assessment: string
  regions_found: number
  malignant: number
  benign: number
  inference_s: number
  _dataUrl?: string | null
  _result: PredictResult
}

export const HistoryStore = {
  list(): LocalHistoryEntry[] {
    if (typeof window === 'undefined') return []
    try { return JSON.parse(localStorage.getItem(LS_HISTORY) || '[]') }
    catch { return [] }
  },
  add(entry: LocalHistoryEntry) {
    const list = HistoryStore.list()
    list.unshift(entry)
    while (list.length > 50) list.pop()
    const trySet = () => localStorage.setItem(LS_HISTORY, JSON.stringify(list))
    try { trySet(); return } catch {}
    for (let i = list.length - 1; i > 0; i--) {
      if (list[i]._dataUrl) { list[i]._dataUrl = null; try { trySet(); return } catch {} }
    }
    while (list.length > 1) {
      list.pop()
      try { trySet(); return } catch {}
    }
  },
  clear() { localStorage.removeItem(LS_HISTORY) },
  get(id: string) { return HistoryStore.list().find(e => e.id === id) },
}

// ── Theme ─────────────────────────────────────────────────────────────────
export const ThemeStore = {
  get(): 'dark' | 'light' {
    if (typeof window === 'undefined') return 'dark'
    return (localStorage.getItem(LS_THEME) as 'dark' | 'light') || 'dark'
  },
  set(t: 'dark' | 'light') {
    localStorage.setItem(LS_THEME, t)
    document.documentElement.classList.toggle('light', t === 'light')
  },
  toggle() { ThemeStore.set(ThemeStore.get() === 'light' ? 'dark' : 'light') },
  init() {
    if (typeof window === 'undefined') return
    document.documentElement.classList.toggle('light', ThemeStore.get() === 'light')
  },
}

// ── Last result cache ─────────────────────────────────────────────────────
export const LastResult = {
  save(result: PredictResult, dataUrl: string) {
    try {
      const safe = { result, dataUrl: dataUrl.length < 4_000_000 ? dataUrl : null }
      localStorage.setItem(LS_LAST_RESULT, JSON.stringify(safe))
    } catch {}
  },
  load(): { result: PredictResult; dataUrl: string | null } | null {
    if (typeof window === 'undefined') return null
    try { return JSON.parse(localStorage.getItem(LS_LAST_RESULT) || 'null') }
    catch { return null }
  },
  clear() { localStorage.removeItem(LS_LAST_RESULT) },
}

// ── API settings ──────────────────────────────────────────────────────────
export const ApiSettings = {
  getBase(): string {
    if (typeof window === 'undefined') return 'http://localhost:8000'
    return (localStorage.getItem(LS_API_BASE) || 'http://localhost:8000').replace(/\/+$/, '')
  },
  setBase(url: string) { localStorage.setItem(LS_API_BASE, url.replace(/\/+$/, '')) },
  getForceMock(): boolean {
    if (typeof window === 'undefined') return false
    return localStorage.getItem(LS_USE_MOCK) === '1'
  },
  setForceMock(v: boolean) { localStorage.setItem(LS_USE_MOCK, v ? '1' : '0') },
}
