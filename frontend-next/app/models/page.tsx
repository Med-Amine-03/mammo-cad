'use client'

import { useState, useEffect } from 'react'
import { t, subscribeLang } from '@/lib/i18n'
import ChartFrame from '@/components/charts/ChartFrame'
import ThresholdSweepChart from '@/components/charts/ThresholdSweepChart'
import RocCurveChart from '@/components/charts/RocCurveChart'
import PrCurveChart from '@/components/charts/PrCurveChart'
import ConfusionMatrix from '@/components/charts/ConfusionMatrix'

function StageHeader({ tag, tagColor, title, subtitle }: { tag: string; tagColor: string; title: string; subtitle: string }) {
  return (
    <div style={{ display: 'flex', alignItems: 'flex-start', gap: 14 }}>
      <span style={{ fontSize: 9.5, fontWeight: 700, padding: '4px 9px', borderRadius: 5, background: `color-mix(in oklab, ${tagColor} 14%, transparent)`, color: tagColor, fontFamily: 'Geist Mono, monospace', letterSpacing: '0.08em', textTransform: 'uppercase', flexShrink: 0, marginTop: 2 }}>
        {tag}
      </span>
      <div>
        <div style={{ fontSize: 16, fontWeight: 600, color: 'var(--txt)', letterSpacing: '-0.2px' }}>{title}</div>
        <div style={{ fontSize: 11.5, color: 'var(--txt3)', marginTop: 3, lineHeight: 1.55 }}>{subtitle}</div>
      </div>
    </div>
  )
}

function HeadlineMetric({ value, label, sub, tone }: { value: string; label: string; sub: string; tone: string }) {
  const color = tone === 'ok' ? 'var(--ok)' : tone === 'warn' ? 'var(--warn)' : 'var(--txt)'
  return (
    <div className="card" style={{ padding: '16px 18px' }}>
      <div className="clbl" style={{ marginBottom: 8 }}>{label}</div>
      <div className="mono" style={{ fontSize: 26, fontWeight: 700, letterSpacing: '-0.6px', color, lineHeight: 1 }}>{value}</div>
      <div style={{ fontSize: 10.5, color: 'var(--txt3)', marginTop: 6, fontFamily: 'Geist Mono, monospace' }}>{sub}</div>
    </div>
  )
}

function PipeStep({ n, icon, color, title, lines }: { n: string; icon: React.ReactNode; color: string; title: string; lines: string[] }) {
  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 8, padding: 14, borderRadius: 10, background: 'var(--bg)', border: '1px solid var(--brd)' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 2 }}>
        <div style={{ width: 30, height: 30, borderRadius: 8, background: `color-mix(in oklab, ${color} 14%, transparent)`, color, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>{icon}</div>
        <span className="mono" style={{ fontSize: 10, color: 'var(--txt3)' }}>{n}</span>
      </div>
      <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--txt)', letterSpacing: '-0.1px' }}>{title}</div>
      <div style={{ fontSize: 10.5, color: 'var(--txt3)', lineHeight: 1.55 }}>
        {lines.map((l, i) => <div key={i}>{l}</div>)}
      </div>
    </div>
  )
}

function PipeArrow() {
  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '0 6px', color: 'var(--txt3)' }}>
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M3 8h10M9 4l4 4-4 4" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round"/></svg>
    </div>
  )
}

function MetricBox({ value, label, tone, big }: { value: string; label: string; tone?: string; big?: boolean }) {
  const color = tone === 'ok' ? 'var(--ok)' : tone === 'warn' ? 'var(--warn)' : tone === 'bad' ? 'var(--bad)' : 'var(--txt)'
  return (
    <div style={{ padding: '12px 14px', background: 'var(--bg)', border: '1px solid var(--brd)', borderRadius: 8, textAlign: 'center' }}>
      <div className="mono" style={{ fontSize: big ? 22 : 16, fontWeight: 700, letterSpacing: '-0.5px', color, lineHeight: 1.1 }}>{value}</div>
      <div className="clbl" style={{ marginTop: 4 }}>{label}</div>
    </div>
  )
}

function DatasetRow({ name, role, roleColor, desc, stats }: { name: string; role: string; roleColor: string; desc: string; stats: { k: string; v: string }[] }) {
  return (
    <div style={{ padding: 14, borderRadius: 9, background: 'var(--bg)', border: '1px solid var(--brd)' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 9, marginBottom: 6 }}>
        <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--txt)' }}>{name}</div>
        <span style={{ fontSize: 9, fontWeight: 700, padding: '2px 7px', borderRadius: 4, background: `color-mix(in oklab, ${roleColor} 12%, transparent)`, color: roleColor, fontFamily: 'Geist Mono, monospace', letterSpacing: '0.05em' }}>{role}</span>
      </div>
      <div style={{ fontSize: 11.5, color: 'var(--txt3)', lineHeight: 1.6, marginBottom: 10 }}>{desc}</div>
      <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap' }}>
        {stats.map(s => (
          <div key={s.k}>
            <div className="clbl" style={{ marginBottom: 2 }}>{s.k}</div>
            <div className="mono" style={{ fontSize: 12, color: 'var(--txt)', fontWeight: 600 }}>{s.v}</div>
          </div>
        ))}
      </div>
    </div>
  )
}

function KvList({ items }: { items: [string, string][] }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column' }}>
      {items.map(([k, v], i) => (
        <div key={i} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '7px 0', borderBottom: i < items.length - 1 ? '1px solid var(--brd)' : 'none' }}>
          <span style={{ fontSize: 11.5, color: 'var(--txt2)' }}>{k}</span>
          <span className="mono" style={{ fontSize: 11, color: 'var(--txt)', fontWeight: 600 }}>{v}</span>
        </div>
      ))}
    </div>
  )
}


export default function ModelsPage() {
  const [, setLang] = useState('')
  useEffect(() => subscribeLang(setLang), [])

  return (
    <div className="fade-in">
      <section className="hero" style={{ paddingBottom: 24 }}>
        <div className="eyebrow"><span className="dot"/>{t('models.eyebrow')}</div>
        <h1 className="h1">{t('models.h1.before')}<em>{t('models.h1.em')}</em>{t('models.h1.after')}</h1>
        <p className="lead">{t('models.lead')}</p>
      </section>

      <div className="container" style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>

        {/* Pipeline */}
        <div className="card" style={{ padding: 20 }}>
          <div className="clbl" style={{ marginBottom: 14 }}>{t('models.pipeline.title')}</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr auto 1fr auto 1fr auto 1fr auto 1fr', alignItems: 'stretch', gap: 0 }}>
            <PipeStep n="1" color="var(--info)" title={t('models.pipeline.s1.t')}
              icon={<svg width="16" height="16" viewBox="0 0 16 16" fill="none"><rect x="2" y="2" width="12" height="12" rx="2" stroke="currentColor" strokeWidth="1.4"/><path d="M5 8h6M8 5v6" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round"/></svg>}
              lines={[t('models.pipeline.s1.a'),t('models.pipeline.s1.b')]}/>
            <PipeArrow/>
            <PipeStep n="2" color="var(--ok)" title={t('models.pipeline.s2.t')}
              icon={<svg width="16" height="16" viewBox="0 0 16 16" fill="none"><rect x="2" y="5" width="12" height="6" rx="1.6" stroke="currentColor" strokeWidth="1.4"/><path d="M5 5V3M11 5V3M5 11v2M11 11v2" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round"/></svg>}
              lines={[t('models.pipeline.s2.a'),t('models.pipeline.s2.b'),t('models.pipeline.s2.c')]}/>
            <PipeArrow/>
            <PipeStep n="3" color="var(--warn)" title={t('models.pipeline.s3.t')}
              icon={<svg width="16" height="16" viewBox="0 0 16 16" fill="none"><circle cx="8" cy="8" r="3" stroke="currentColor" strokeWidth="1.4"/><circle cx="8" cy="8" r="6" stroke="currentColor" strokeWidth="1.2" strokeDasharray="2.5 2.5"/></svg>}
              lines={[t('models.pipeline.s3.a'),t('models.pipeline.s3.b'),t('models.pipeline.s3.c')]}/>
            <PipeArrow/>
            <PipeStep n="4" color="var(--bad)" title={t('models.pipeline.s4.t')}
              icon={<svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M8 2C5.2 2 3 4.2 3 7c0 1.8 1 3.4 2.4 4.3V13h5v-1.7C11.9 10.4 13 8.8 13 7c0-2.8-2.2-5-5-5Z" stroke="currentColor" strokeWidth="1.3" fill="none"/><path d="M6 13h4M7 11V9M9 11V9" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round"/></svg>}
              lines={[t('models.pipeline.s4.a'),t('models.pipeline.s4.b'),t('models.pipeline.s4.c')]}/>
            <PipeArrow/>
            <PipeStep n="5" color="var(--info)" title={t('models.pipeline.s5.t')}
              icon={<svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M2 12V6l3-2 2 2 3-3 2 2v7H2Z" stroke="currentColor" strokeWidth="1.4" fill="none" strokeLinejoin="round"/></svg>}
              lines={[t('models.pipeline.s5.a'),t('models.pipeline.s5.b'),t('models.pipeline.s5.c')]}/>
          </div>
        </div>

        {/* Headline metrics */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 14 }}>
          <HeadlineMetric value="0.9642" label={t('models.head.segAuc')} sub={t('models.head.segAuc.s')} tone="ok"/>
          <HeadlineMetric value="0.775"  label={t('models.head.dice')}   sub={t('models.head.dice.s')}   tone="ok"/>
          <HeadlineMetric value="0.7812" label={t('models.head.clsAuc')} sub={t('models.head.clsAuc.s')} tone="warn"/>
          <HeadlineMetric value="71.0%"  label={t('models.head.acc')}    sub={t('models.head.acc.s')}    tone="warn"/>
        </div>

        {/* Stage 1 */}
        <div className="card" style={{ padding: 20 }}>
          <StageHeader tag={t('models.s1.tag')} tagColor="var(--ok)" title={t('models.s1.title')} subtitle={t('models.s1.subtitle')}/>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 14, marginTop: 14 }}>
            <ThresholdSweepChart/>
            <RocCurveChart auc={0.9642} color="#60a5fa" title={t('models.charts.rocSeg')} sub={t('models.charts.rocSeg.s')}/>
            <PrCurveChart  ap={0.8009}  color="#f87171" title={t('models.charts.pr')}     sub={t('models.charts.pr.s')}/>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6,1fr)', gap: 8, marginTop: 14 }}>
            <MetricBox value="0.964" label={t('models.metric.rocAuc')} tone="ok" big/>
            <MetricBox value="0.801" label={t('models.metric.prAuc')}  tone="ok" big/>
            <MetricBox value="0.775" label={t('models.metric.dice')}   tone="ok"/>
            <MetricBox value="0.632" label={t('models.metric.iou')}/>
            <MetricBox value="0.766" label={t('models.metric.sens')}/>
            <MetricBox value="0.50"  label={t('models.metric.thr')}/>
          </div>
          <div style={{ marginTop: 14, padding: '11px 13px', background: 'var(--ab)', borderRadius: 8, fontSize: 11, color: 'var(--txt3)', lineHeight: 1.65, border: '1px solid var(--brd)' }}>
            {t('models.s1.note')}
          </div>
        </div>

        {/* Stage 2 */}
        <div className="card" style={{ padding: 20 }}>
          <StageHeader tag={t('models.s2.tag')} tagColor="var(--warn)" title={t('models.s2.title')} subtitle={t('models.s2.subtitle')}/>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14, marginTop: 14 }}>
            <ChartFrame title={t('models.charts.testRocPr')} sub={t('models.charts.testRocPr.s')}>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src="/assets/test_roc_pr.png" alt="Test set ROC and PR curves"
                style={{ width: '100%', height: 'auto', display: 'block', borderRadius: 6, filter: 'invert(0.92) hue-rotate(180deg) saturate(0.85) brightness(0.95)' }}
                onError={e => { (e.target as HTMLImageElement).style.display = 'none' }}
              />
            </ChartFrame>
            <ConfusionMatrix title={t('models.charts.cm')} auc={0.7812} tn={362} fp={113} fn={115} tp={197}/>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6,1fr)', gap: 8, marginTop: 14 }}>
            <MetricBox value="0.7812" label={t('models.metric.rocAuc')} tone="warn" big/>
            <MetricBox value="0.7183" label={t('models.metric.ap')}     tone="warn" big/>
            <MetricBox value="71.0%"  label={t('models.metric.acc')}/>
            <MetricBox value="63.1%"  label={t('models.metric.sens')}/>
            <MetricBox value="76.2%"  label={t('models.metric.spec')}/>
            <MetricBox value="0.53"   label={t('models.metric.opThr')}/>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14, marginTop: 14 }}>
            <ChartFrame title={t('models.charts.s2curves')} sub={t('models.charts.s2curves.s')}>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src="/assets/stage2_curves.png" alt="Stage 2 training curves"
                style={{ width: '100%', height: 'auto', display: 'block', borderRadius: 6, filter: 'invert(0.92) hue-rotate(180deg) saturate(0.85) brightness(0.95)' }}
                onError={e => { (e.target as HTMLImageElement).style.display = 'none' }}
              />
            </ChartFrame>
            <ChartFrame title={t('models.charts.s2ft')} sub={t('models.charts.s2ft.s')}>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src="/assets/stage2_finetune_curves.png" alt="Stage 2 fine-tuning curves"
                style={{ width: '100%', height: 'auto', display: 'block', borderRadius: 6, filter: 'invert(0.92) hue-rotate(180deg) saturate(0.85) brightness(0.95)' }}
                onError={e => { (e.target as HTMLImageElement).style.display = 'none' }}
              />
            </ChartFrame>
          </div>
          <div style={{ marginTop: 14 }}>
            <ChartFrame title={t('models.charts.calcMass')} sub={t('models.charts.calcMass.s')}>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src="/assets/calc_vs_mass.png" alt="Calcifications vs Masses"
                style={{ width: '100%', height: 'auto', display: 'block', borderRadius: 6 }}
                onError={e => { (e.target as HTMLImageElement).style.display = 'none' }}
              />
            </ChartFrame>
          </div>
        </div>

        {/* Sample inferences */}
        <div className="card" style={{ padding: 20 }}>
          <StageHeader tag={t('models.qual.tag')} tagColor="var(--info)" title={t('models.qual.title')} subtitle={t('models.qual.subtitle')}/>
          <div style={{ marginTop: 14, background: 'var(--bg)', border: '1px solid var(--brd)', borderRadius: 10, padding: 10, maxHeight: 560, overflowY: 'auto' }} className="nice-scroll">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src="/assets/multi_image_comparison.png" alt="Multi-image inference comparison"
              style={{ width: '100%', display: 'block', borderRadius: 6 }}
              onError={e => { (e.target as HTMLImageElement).style.display = 'none' }}
            />
          </div>
          <div style={{ marginTop: 10, fontSize: 10.5, color: 'var(--txt3)', display: 'flex', gap: 18, flexWrap: 'wrap' }}>
            <span><span style={{ color: '#34d399' }}>■</span> {t('models.qual.legend.b')}</span>
            <span><span style={{ color: '#f87171' }}>■</span> {t('models.qual.legend.m')}</span>
            <span style={{ marginLeft: 'auto', fontFamily: 'Geist Mono, monospace' }}>{t('models.qual.legend.n')}</span>
          </div>
        </div>

        {/* Datasets */}
        <div className="card" style={{ padding: 18 }}>
          <div className="clbl" style={{ marginBottom: 12 }}>{t('models.datasets')}</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            <DatasetRow name="INbreast" role={t('models.dataset.inbreast.role')} roleColor="var(--ok)" desc={t('models.dataset.inbreast.desc')}
              stats={[{ k:t('models.dataset.images'),v:'410'},{k:t('models.dataset.cases'),v:'115'},{k:t('models.dataset.format'),v:'DICOM'},{k:t('models.dataset.split'),v:'10-fold CV'}]}/>
            <DatasetRow name="CBIS-DDSM" role={t('models.dataset.cbis.role')} roleColor="var(--warn)" desc={t('models.dataset.cbis.desc')}
              stats={[{k:t('models.dataset.cases'),v:'1,566'},{k:t('models.dataset.rois'),v:'~3.1k'},{k:t('models.dataset.test'),v:'787'},{k:t('models.dataset.license'),v:'Public'}]}/>
          </div>
        </div>

        {/* Hyperparams */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
          <div className="card" style={{ padding: 18 }}>
            <div className="clbl" style={{ marginBottom: 12, color: 'var(--ok)' }}>{t('models.train.unet')}</div>
            <KvList items={[
              [t('models.train.optimizer'), 'AdamW · lr=1e-3'],
              [t('models.train.scheduler'), 'CosineAnnealing'],
              [t('models.train.batch'),     '16'],
              [t('models.train.loss'),      'Dice + OHEM-BCE'],
              [t('models.train.aug'),       'Flip · Rotate · Elastic'],
              [t('models.train.cv'),        '10-fold (patient-level)'],
              [t('models.train.hw'),        'NVIDIA T4 · 16 GB'],
              [t('models.train.bestThr'),   '0.50'],
            ]}/>
          </div>
          <div className="card" style={{ padding: 18 }}>
            <div className="clbl" style={{ marginBottom: 12, color: 'var(--warn)' }}>{t('models.train.effb3')}</div>
            <KvList items={[
              [t('models.train.optimizer'), 'AdamW · lr_head=1e-4 · lr_bb=1e-5'],
              [t('models.train.scheduler'), 'Step decay (head) + cosine (bb)'],
              [t('models.train.batch'),     '32'],
              [t('models.train.stages'),    'Stage 1: head · Stage 2: full fine-tune'],
              [t('models.train.loss'),      'CE + label smoothing 0.1'],
              [t('models.train.sampler'),   'WeightedRandom'],
              [t('models.train.tta'),       '16 passes (flip/rot)'],
              [t('models.train.opThr'),     '0.53 (sens-target = 0.80)'],
            ]}/>
          </div>
        </div>

        {/* Tech stack */}
        <div className="card" style={{ padding: 18 }}>
          <div className="clbl" style={{ marginBottom: 12 }}>{t('models.stack')}</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
            {[
              {n:'PyTorch',c:'#ee4c2c'},{n:'FastAPI',c:'#009688'},{n:'EfficientNet-B3',c:'#f7931e'},
              {n:'U-Net',c:'#60a5fa'},{n:'Grad-CAM++',c:'#f87171'},{n:'Python 3.10',c:'#34d399'},
              {n:'OpenCV',c:'#fbbf24'},{n:'NumPy / SciPy',c:'#60a5fa'},{n:'scikit-learn',c:'#f87171'},
              {n:'pydicom',c:'#a3a5ab'},{n:'Pillow',c:'#34d399'},{n:'uvicorn',c:'#a78bfa'},{n:'ngrok',c:'#1F3D6F'},
            ].map(tag => (
              <span key={tag.n} style={{ display: 'inline-flex', alignItems: 'center', gap: 6, padding: '6px 11px', borderRadius: 7, border: '1px solid var(--brd2)', background: 'var(--bg)', fontSize: 11.5, fontWeight: 500, color: 'var(--txt2)' }}>
                <span style={{ width: 6, height: 6, borderRadius: '50%', background: tag.c }}/>
                {tag.n}
              </span>
            ))}
          </div>
        </div>

      </div>
    </div>
  )
}
