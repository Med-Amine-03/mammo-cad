'use client'

import ChartFrame from './ChartFrame'

const DEFAULT_POINTS: [number,number][] = [
  [0,0],[0.005,0.85],[0.01,0.91],[0.02,0.93],[0.04,0.945],[0.08,0.955],
  [0.15,0.965],[0.25,0.975],[0.4,0.985],[0.6,0.992],[0.8,0.997],[1,1],
]

interface Props {
  auc: number
  color?: string
  title?: string
  sub?: string
  points?: [number,number][]
}

export default function RocCurveChart({ auc, color='#60a5fa', title='ROC Curve', sub, points }: Props) {
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
        <line x1={xf(0)} y1={yf(0)} x2={xf(1)} y2={yf(1)} stroke="var(--txt3)" strokeDasharray="3 3" opacity="0.5"/>
        <path d={path} fill="none" stroke={color} strokeWidth="1.8"/>
        <text x={W-18} y={P+12} fontSize="10" fill={color} textAnchor="end" fontFamily="Geist Mono" fontWeight="600">
          AUC = {auc.toFixed(4)}
        </text>
        <text x={W/2} y={H-6} fontSize="9" fill="var(--txt3)" textAnchor="middle">FPR</text>
        <text x={10} y={H/2} fontSize="9" fill="var(--txt3)" textAnchor="middle" transform={`rotate(-90 10 ${H/2})`}>TPR</text>
      </svg>
    </ChartFrame>
  )
}
