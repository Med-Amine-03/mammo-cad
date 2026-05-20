'use client'

import ChartFrame from './ChartFrame'

const DATA = [
  { t:0.10, iou:0.560, dice:0.720, sens:0.857 },
  { t:0.15, iou:0.585, dice:0.740, sens:0.842 },
  { t:0.20, iou:0.602, dice:0.752, sens:0.831 },
  { t:0.25, iou:0.611, dice:0.760, sens:0.820 },
  { t:0.30, iou:0.619, dice:0.766, sens:0.808 },
  { t:0.35, iou:0.624, dice:0.769, sens:0.797 },
  { t:0.40, iou:0.628, dice:0.772, sens:0.787 },
  { t:0.45, iou:0.630, dice:0.774, sens:0.776 },
  { t:0.50, iou:0.632, dice:0.775, sens:0.766 },
  { t:0.55, iou:0.631, dice:0.775, sens:0.738 },
  { t:0.60, iou:0.629, dice:0.773, sens:0.722 },
  { t:0.65, iou:0.624, dice:0.770, sens:0.706 },
  { t:0.70, iou:0.619, dice:0.766, sens:0.681 },
  { t:0.75, iou:0.606, dice:0.756, sens:0.652 },
  { t:0.80, iou:0.591, dice:0.745, sens:0.621 },
  { t:0.85, iou:0.572, dice:0.728, sens:0.578 },
  { t:0.90, iou:0.539, dice:0.700, sens:0.498 },
]

export default function ThresholdSweepChart() {
  const W=420, H=240, PL=42, PR=14, PT=14, PB=30
  const xf = (t: number) => PL + ((t - 0.10)/(0.90 - 0.10)) * (W-PL-PR)
  const yf = (v: number) => PT + (1 - (v - 0.50)/(0.90 - 0.50)) * (H-PT-PB)
  const series = (key: 'iou'|'dice'|'sens') =>
    DATA.map((d, i) => `${i ? 'L' : 'M'}${xf(d.t)} ${yf(d[key])}`).join(' ')

  return (
    <ChartFrame title="Threshold sweep" sub="IoU · Dice · Sensitivity vs. threshold (best @ 0.50)">
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto' }}>
        {[0.5,0.6,0.7,0.8,0.9].map(v => (
          <g key={v}>
            <line x1={PL} x2={W-PR} y1={yf(v)} y2={yf(v)} stroke="var(--brd)" strokeDasharray="2 3"/>
            <text x={PL-6} y={yf(v)+3} fontSize="9" textAnchor="end" fill="var(--txt3)" fontFamily="Geist Mono">{v.toFixed(1)}</text>
          </g>
        ))}
        {[0.1,0.3,0.5,0.7,0.9].map(t => (
          <text key={t} x={xf(t)} y={H-12} fontSize="9" textAnchor="middle" fill="var(--txt3)" fontFamily="Geist Mono">{t.toFixed(1)}</text>
        ))}
        <line x1={xf(0.5)} x2={xf(0.5)} y1={PT} y2={H-PB} stroke="#60a5fa" strokeDasharray="3 3" opacity="0.55"/>
        <path d={series('iou')}  fill="none" stroke="#60a5fa" strokeWidth="1.6"/>
        <path d={series('dice')} fill="none" stroke="#f87171" strokeWidth="1.6"/>
        <path d={series('sens')} fill="none" stroke="#34d399" strokeWidth="1.6"/>
        {DATA.map((d,i) => (
          <g key={i}>
            <circle cx={xf(d.t)} cy={yf(d.iou)}  r="2" fill="#60a5fa"/>
            <circle cx={xf(d.t)} cy={yf(d.dice)} r="2" fill="#f87171"/>
            <circle cx={xf(d.t)} cy={yf(d.sens)} r="2" fill="#34d399"/>
          </g>
        ))}
        <g fontSize="9" fontFamily="Geist Mono" fill="var(--txt2)">
          <rect x={PL+8} y={PT+6} width="106" height="46" fill="var(--bg)" stroke="var(--brd)" rx="4"/>
          <circle cx={PL+18} cy={PT+16} r="3" fill="#60a5fa"/><text x={PL+26} y={PT+19}>IoU</text>
          <circle cx={PL+18} cy={PT+28} r="3" fill="#f87171"/><text x={PL+26} y={PT+31}>Dice</text>
          <circle cx={PL+18} cy={PT+40} r="3" fill="#34d399"/><text x={PL+26} y={PT+43}>Sensitivity</text>
        </g>
      </svg>
    </ChartFrame>
  )
}
