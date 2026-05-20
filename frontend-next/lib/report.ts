import type { Region } from './types'
import { fmt } from './utils'

export interface ReportData {
  id:                 string
  filename:           string
  overall_assessment: string
  regions_found:      number
  malignant:          number
  benign:             number
  inference_s:        number
  results:            Region[]
  image_size?:        [number, number] | null
}

export function generateHtmlReport(data: ReportData): string {
  const now     = new Date()
  const dateStr = now.toLocaleDateString('fr-FR', { day: '2-digit', month: 'long', year: 'numeric' })
  const timeStr = now.toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' })

  const assessLabel = (() => {
    const a = (data.overall_assessment || '').toUpperCase()
    if (a === 'MALIGNANT' || a === 'SUSPICIOUS')
      return { text: 'SUSPECT — MALIGNE', color: '#dc2626', bg: '#fef2f2', border: '#fecaca' }
    if (a === 'BENIGN')
      return { text: 'BÉNIN', color: '#d97706', bg: '#fffbeb', border: '#fde68a' }
    return { text: a || 'INDÉTERMINÉ', color: '#6b7280', bg: '#f9fafb', border: '#e5e7eb' }
  })()

  const regionsHtml = (data.results || []).map(r => {
    const isMal      = r.label === 'MALIGNANT'
    const [x1, y1, x2, y2] = r.location.map(n => Math.round(n))
    const pct        = (r.probability  * 100).toFixed(1)
    const unc        = (r.uncertainty  * 100).toFixed(2)
    const barColor   = isMal ? '#dc2626' : '#d97706'
    const rowBg      = isMal ? '#fff5f5' : '#fffdf0'
    const labelText  = isMal ? 'MALIGNE'  : 'BÉNIGNE'
    const labelColor = isMal ? '#dc2626'  : '#d97706'
    const labelBg    = isMal ? '#fef2f2'  : '#fffbeb'
    return `
      <div style="margin-bottom:12px;border:1px solid #e5e7eb;border-radius:10px;overflow:hidden;background:${rowBg}">
        <div style="padding:10px 14px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid #e5e7eb;background:#fff">
          <span style="font-weight:700;font-size:14px;color:#111">Région R${r.region_id}</span>
          <span style="font-size:11px;font-weight:700;padding:3px 10px;border-radius:999px;background:${labelBg};color:${labelColor};border:1px solid ${labelColor}30">${labelText}</span>
        </div>
        <div style="padding:12px 14px;display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px">
          <div>
            <div style="font-size:10px;color:#6b7280;text-transform:uppercase;letter-spacing:.06em;font-weight:600;margin-bottom:5px">Probabilité</div>
            <div style="font-size:22px;font-weight:800;color:${barColor};font-family:monospace;line-height:1">${pct}%</div>
            <div style="margin-top:6px;height:6px;border-radius:999px;background:#e5e7eb;overflow:hidden">
              <div style="height:100%;width:${pct}%;background:${barColor};border-radius:999px"></div>
            </div>
          </div>
          <div>
            <div style="font-size:10px;color:#6b7280;text-transform:uppercase;letter-spacing:.06em;font-weight:600;margin-bottom:5px">Incertitude</div>
            <div style="font-size:18px;font-weight:700;color:#374151;font-family:monospace">±${unc}%</div>
          </div>
          <div>
            <div style="font-size:10px;color:#6b7280;text-transform:uppercase;letter-spacing:.06em;font-weight:600;margin-bottom:5px">Localisation (px)</div>
            <div style="font-size:11px;color:#374151;font-family:monospace;line-height:1.6">
              Haut-gauche : (${x1}, ${y1})<br>
              Bas-droite&nbsp; : (${x2}, ${y2})<br>
              Taille&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; : ${x2 - x1} × ${y2 - y1} px
            </div>
          </div>
        </div>
      </div>`
  }).join('')

  const kpis = [
    ['Régions détectées', String(data.regions_found), '#111'],
    ['Régions suspectes', String(data.malignant), data.malignant > 0 ? '#dc2626' : '#6b7280'],
    ['Régions bénignes',  String(data.benign),    data.benign   > 0 ? '#d97706' : '#6b7280'],
    ["Temps d'inférence", fmt.time(data.inference_s), '#111'],
  ].map(([label, val, color]) => `
    <div style="padding:14px 16px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;text-align:center">
      <div style="font-size:26px;font-weight:800;color:${color};font-family:monospace;line-height:1.1">${val}</div>
      <div style="font-size:10px;color:#6b7280;margin-top:5px;font-weight:500">${label}</div>
    </div>`).join('')

  return `<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8"/>
  <title>Rapport MammoCAD — ${data.filename}</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0 }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f3f4f6; color: #111; }
    .page { max-width: 760px; margin: 32px auto; background: #fff; border-radius: 14px; overflow: hidden; box-shadow: 0 4px 24px rgba(0,0,0,.10); }
    @media print {
      body { background: #fff }
      .page { max-width: 100%; margin: 0; border-radius: 0; box-shadow: none }
      .no-print { display: none }
    }
  </style>
</head>
<body>
  <div class="page">
    <div style="background:linear-gradient(135deg,#0f172a 0%,#1e293b 100%);padding:28px 32px;color:#fff">
      <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:16px;flex-wrap:wrap">
        <div>
          <div style="font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:#94a3b8;margin-bottom:8px;font-weight:600">Rapport d'analyse CAD — Mammographie</div>
          <div style="font-size:22px;font-weight:800;letter-spacing:-.4px">MammoCAD</div>
        </div>
        <div style="text-align:right">
          <div style="font-size:12px;color:#94a3b8">${dateStr} à ${timeStr}</div>
          <div style="font-size:11px;color:#64748b;margin-top:4px;font-family:monospace">ID ${data.id}</div>
        </div>
      </div>
    </div>

    <div style="padding:28px 32px">
      <div style="margin-bottom:24px;padding:14px 16px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px">
        <div style="font-size:10px;color:#6b7280;text-transform:uppercase;letter-spacing:.06em;font-weight:600;margin-bottom:4px">Fichier analysé</div>
        <div style="font-size:15px;font-weight:700;color:#111">${data.filename}</div>
        ${data.image_size ? `<div style="font-size:11px;color:#6b7280;margin-top:2px;font-family:monospace">Dimensions : ${data.image_size[0]} × ${data.image_size[1]} px</div>` : ''}
      </div>

      <div style="margin-bottom:24px;padding:18px 20px;background:${assessLabel.bg};border:2px solid ${assessLabel.border};border-radius:12px">
        <div style="font-size:10px;color:#6b7280;text-transform:uppercase;letter-spacing:.06em;font-weight:600;margin-bottom:8px">Évaluation globale</div>
        <div style="font-size:24px;font-weight:900;color:${assessLabel.color};letter-spacing:-.5px">${assessLabel.text}</div>
      </div>

      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:28px">
        ${kpis}
      </div>

      <div style="margin-bottom:28px">
        <div style="font-size:13px;font-weight:700;color:#111;margin-bottom:12px;padding-bottom:8px;border-bottom:2px solid #f1f5f9">
          Détail des régions détectées (${(data.results || []).length})
        </div>
        ${regionsHtml || '<div style="padding:20px;text-align:center;color:#9ca3af;font-size:13px">Aucune région détectée</div>'}
      </div>

      <div style="padding-top:20px;border-top:1px solid #e5e7eb;display:flex;justify-content:space-between;align-items:center">
        <span style="font-size:11px;font-weight:700;color:#94a3b8;letter-spacing:.04em">MammoCAD</span>
        <span style="font-size:10px;color:#d1d5db">${dateStr}</span>
      </div>
    </div>

    <div class="no-print" style="padding:16px 32px;background:#f8fafc;border-top:1px solid #e5e7eb;text-align:right">
      <button onclick="window.print()" style="padding:9px 20px;background:#0f172a;color:#fff;border:none;border-radius:8px;font-size:13px;font-weight:600;cursor:pointer">
        🖨️ Imprimer / Enregistrer en PDF
      </button>
    </div>
  </div>
</body>
</html>`
}

export function triggerDownload(html: string, filename: string) {
  const blob = new Blob([html], { type: 'text/html;charset=utf-8' })
  const url  = URL.createObjectURL(blob)
  const a    = document.createElement('a')
  a.href = url; a.download = filename
  document.body.appendChild(a); a.click(); a.remove()
  URL.revokeObjectURL(url)
}
