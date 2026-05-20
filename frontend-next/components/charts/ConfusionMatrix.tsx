'use client'

import ChartFrame from './ChartFrame'

interface CmCellProps {
  value: number; pct: string; kind: 'good'|'bad'; intensity: number; label: string
}

function CmCell({ value, pct, kind, intensity, label }: CmCellProps) {
  const isGood = kind === 'good'
  return (
    <div style={{
      padding: '18px 14px',
      borderRadius: 8,
      background: isGood
        ? `color-mix(in oklab, var(--ok) ${intensity*38}%, var(--bg))`
        : `color-mix(in oklab, var(--bad) ${intensity*38}%, var(--bg))`,
      border: `1px solid ${isGood
        ? 'color-mix(in oklab, var(--ok) 35%, transparent)'
        : 'color-mix(in oklab, var(--bad) 35%, transparent)'}`,
      textAlign: 'center',
    }}>
      <div className="mono" style={{ fontSize: 9, color: 'var(--txt3)', marginBottom: 4, letterSpacing: '0.08em' }}>{label}</div>
      <div className="mono" style={{ fontSize: 24, fontWeight: 700, color: 'var(--txt)', lineHeight: 1 }}>{value}</div>
      <div className="mono" style={{ fontSize: 11, color: isGood ? 'var(--ok)' : 'var(--bad)', marginTop: 4, fontWeight: 600 }}>{pct}%</div>
    </div>
  )
}

interface Props {
  tn: number; fp: number; fn: number; tp: number
  auc: number; title?: string
}

export default function ConfusionMatrix({ tn, fp, fn, tp, auc, title }: Props) {
  const total = tn + fp + fn + tp
  const pct = (v: number) => ((v / total) * 100).toFixed(1)
  const maxVal = Math.max(tn, fp, fn, tp)

  return (
    <ChartFrame title={title || 'Confusion matrix'} sub={`AUC = ${auc.toFixed(4)} · n = ${total}`}>
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'auto 1fr 1fr',
        gridTemplateRows: 'auto 1fr 1fr',
        gap: 6, padding: '4px 8px 8px',
      }}>
        <div/>
        <div className="clbl" style={{ textAlign: 'center' }}>Pred: Benign</div>
        <div className="clbl" style={{ textAlign: 'center' }}>Pred: Malignant</div>

        <div className="clbl" style={{
          display: 'flex', alignItems: 'center', justifyContent: 'flex-end',
          paddingRight: 6, writingMode: 'vertical-rl', transform: 'rotate(180deg)',
        }}>True: Benign</div>
        <CmCell value={tn} pct={pct(tn)} kind="good" intensity={tn/maxVal} label="TN"/>
        <CmCell value={fp} pct={pct(fp)} kind="bad"  intensity={fp/maxVal} label="FP"/>

        <div className="clbl" style={{
          display: 'flex', alignItems: 'center', justifyContent: 'flex-end',
          paddingRight: 6, writingMode: 'vertical-rl', transform: 'rotate(180deg)',
        }}>True: Malignant</div>
        <CmCell value={fn} pct={pct(fn)} kind="bad"  intensity={fn/maxVal} label="FN"/>
        <CmCell value={tp} pct={pct(tp)} kind="good" intensity={tp/maxVal} label="TP"/>
      </div>
    </ChartFrame>
  )
}
