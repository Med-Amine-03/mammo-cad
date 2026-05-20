export const fmt = {
  pct(v: number, digits = 1) { return (v * 100).toFixed(digits) + '%' },
  prob(v: number) { return v.toFixed(3) },
  unc(v: number) { return '±' + v.toFixed(3) },
  bbox(loc: number[]) { return `[${loc.map(n => Math.round(n)).join(', ')}]` },
  time(s: number) { return s < 1 ? `${Math.round(s * 1000)}ms` : `${s.toFixed(2)}s` },
  bytes(b?: number | null) {
    if (b == null) return '—'
    if (b < 1024) return `${b} B`
    if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`
    return `${(b / 1024 / 1024).toFixed(2)} MB`
  },
  date(ts: string | number) {
    const d = new Date(ts)
    return d.toLocaleString(undefined, { month: 'short', day: '2-digit', hour: '2-digit', minute: '2-digit' })
  },
  shortId(id: string) { return (id || '').slice(0, 10) },
}

export function highlightJson(obj: unknown): string {
  const json = JSON.stringify(obj, null, 2)
  return json
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/("(?:\\.|[^"\\])*")(\s*:)/g, '<span class="jk">$1</span>$2')
    .replace(/:\s*("(?:\\.|[^"\\])*")/g, ': <span class="jv">$1</span>')
    .replace(/:\s*(true|false|null)/g, ': <span class="jp">$1</span>')
    .replace(/:\s*(-?\d+\.?\d*)/g, ': <span class="js">$1</span>')
}

export function mulberry(seed: number) {
  let t = seed >>> 0
  return function () {
    t = (t + 0x6D2B79F5) >>> 0
    let r = Math.imul(t ^ (t >>> 15), 1 | t)
    r = (r + Math.imul(r ^ (r >>> 7), 61 | r)) ^ r
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296
  }
}

export const sleep = (ms: number) => new Promise(r => setTimeout(r, ms))

export function fileToDataUrl(file: File): Promise<string> {
  return new Promise((res, rej) => {
    const fr = new FileReader()
    fr.onload = () => res(fr.result as string)
    fr.onerror = rej
    fr.readAsDataURL(file)
  })
}

export function imageDims(src: string): Promise<{ w: number; h: number }> {
  return new Promise((res, rej) => {
    const img = new Image()
    img.onload = () => res({ w: img.naturalWidth, h: img.naturalHeight })
    img.onerror = rej
    img.src = src
  })
}
