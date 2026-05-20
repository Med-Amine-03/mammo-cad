'use client'

import ChartFrame from './ChartFrame'

const DEFAULT_POINTS: [number,number][] = [
  [0,1],[0.05,0.99],[0.1,0.97],[0.15,0.945],[0.2,0.93],[0.3,0.915],[0.4,0.895],
  [0.5,0.875],[0.6,0.85],[0.7,0.81],[0.8,0.74],[0.85,0.65],[0.9,0.45],[0.95,0.18],[1,0.05],
]

interface Props {
  ap: number
  color?: string
  title?: string
  sub?: string
  points?: [number,number][]
}

export default function PrCurveChart({ ap, color='#f87171', title='Precision-Recall', sub, points }: Props) {
  const W=300, H=240, P=36
  const pts = points || DEFAULT_POINTS
  const xf = (v: number) => P + v*(W-P-12)
  const yf = (v: number) => H-P - v*(H-P-12)
  const path = pts.map((p,i) => `${i?'L':'M'}${xf(p[0])} ${yf(p[1])}`).join(' ')

  return (
    <ChartFrame title={title} sub={sub}>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto' }}>
        {[0,0.2,0.4,0.6,0.8,1].map(v => (
          <g key={v}>
            <line x1={P} x2={W-12} y1={yf(v)} y2={yf(v)} stroke="var(--brd)" strokeDasharray="2 3"/>
            <text x={P-5} y={yf(v)+3} fontSize="9" textAnchor="end" fill="var(--txt3)" fontFamily="Geist Mono">{v.toFixed(1)}</text>
            <text x={xf(v)} y={H-P+14} fontSize="9" textAnchor="middle" fill="var(--txt3)" fontFamily="Geist Mono">{v.toFixed(1)}</text>
          </g>
        ))}
        <path d={path} fill="none" stroke={color} strokeWidth="1.8"/>
        <text x={W-18} y={P+12} fontSize="10" fill={color} textAnchor="end" fontFamily="Geist Mono" fontWeight="600">
          AP = {ap.toFixed(4)}
        </text>
        <text x={W/2} y={H-6} fontSize="9" fill="var(--txt3)" textAnchor="middle">Recall</text>
        <text x={10} y={H/2} fontSize="9" fill="var(--txt3)" textAnchor="middle" transform={`rotate(-90 10 ${H/2})`}>Precision</text>
      </svg>
    </ChartFrame>
  )
}
