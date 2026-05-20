'use client'

interface Props {
  title: string
  sub?: string
  children: React.ReactNode
}

export default function ChartFrame({ title, sub, children }: Props) {
  return (
    <div style={{
      background: 'var(--bg)',
      border: '1px solid var(--brd)',
      borderRadius: 10,
      padding: 14,
      display: 'flex', flexDirection: 'column', gap: 10,
    }}>
      <div>
        <div style={{ fontSize: 12.5, fontWeight: 600, color: 'var(--txt)', marginBottom: 2 }}>{title}</div>
        {sub && <div style={{ fontSize: 10.5, color: 'var(--txt3)', fontFamily: 'Geist Mono, monospace' }}>{sub}</div>}
      </div>
      <div>{children}</div>
    </div>
  )
}
