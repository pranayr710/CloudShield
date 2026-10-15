import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { Chapter } from '@/components/chapter'
import { Slider } from '@/components/ui/slider'
import { gsap, prefersReducedMotion } from '@/lib/gsap'
import { CLASS_COLOR, D, bytes, int, pct } from '@/lib/data'
import { cn } from '@/lib/utils'

const R = D.regression
const FLOWS = [...R.flows].sort((a, b) => a.actual - b.actual)
const W = R.whatIf
const MAX_LOG = 7 // gauge runs from 1 byte to 10 MB
const lg = (v: number) => Math.min(MAX_LOG, Math.log10(Math.max(0, v) + 1))

/** A number that counts to its new value instead of jumping. */
function Counter({ value, format, className }: { value: number; format: (v: number) => string; className?: string }) {
  const ref = useRef<HTMLSpanElement>(null)
  const state = useRef<{ v: number } | null>(null)
  // the text is written here, not rendered by React, so the tween and React never touch the same node
  useLayoutEffect(() => {
    const el = ref.current!
    if (!state.current || prefersReducedMotion()) {
      state.current = { v: value }
      el.textContent = format(value)
      return
    }
    const t = gsap.to(state.current, { v: value, duration: 0.8, ease: 'expo.out', onUpdate: () => { el.textContent = format(state.current!.v) } })
    return () => { t.kill() }
  }, [value, format])
  return <span ref={ref} className={className} />
}

/** Half-circle gauge on a log scale: needle for the prediction, a tick for the truth, a band for the trees' spread. */
function Gauge({ pred, lo, hi, actual }: { pred: number; lo: number; hi: number; actual: number | null }) {
  const cx = 160, cy = 160, r = 130
  const angle = (v: number) => Math.PI * (1 - lg(v) / MAX_LOG)
  const pt = (v: number, rr = r) => [cx + rr * Math.cos(angle(v)), cy - rr * Math.sin(angle(v))]
  const arc = (a: number, b: number, rr: number) => {
    const [x1, y1] = pt(a, rr), [x2, y2] = pt(b, rr)
    return `M${x1} ${y1} A${rr} ${rr} 0 0 1 ${x2} ${y2}`
  }
  const needle = useRef<SVGGElement>(null)
  const deg = 180 - (lg(pred) / MAX_LOG) * 180

  useEffect(() => {
    gsap.to(needle.current, { rotation: -deg, svgOrigin: `${cx} ${cy}`, duration: prefersReducedMotion() ? 0 : 1, ease: 'elastic.out(1, 0.6)' })
  }, [deg])

  const ticks = [0, 1, 2, 3, 4, 5, 6, 7]
  return (
    <svg viewBox="0 0 320 185" className="w-full max-w-[420px]" role="img" aria-label={`Gauge: predicted ${bytes(pred)}`}>
      <path d={arc(0, 1e7, r)} fill="none" stroke="rgba(22,20,15,0.12)" strokeWidth="14" strokeLinecap="round" />
      <path d={arc(0, pred, r)} fill="none" stroke="url(#g-fill)" strokeWidth="14" strokeLinecap="round" style={{ transition: 'all 0.6s' }} />
      {hi > lo && <path d={arc(lo, hi, r + 16)} fill="none" stroke="#0f6e5a" strokeOpacity="0.5" strokeWidth="4" strokeLinecap="round" />}
      <defs>
        <linearGradient id="g-fill" x1="0" x2="1">
          <stop offset="0%" stopColor="#0f6e5a" />
          <stop offset="70%" stopColor="#2f5d9e" />
          <stop offset="100%" stopColor="#e5481c" />
        </linearGradient>
      </defs>
      {ticks.map((t) => {
        const v = 10 ** t - 1
        const [x1, y1] = pt(v, r - 14), [x2, y2] = pt(v, r - 22), [tx, ty] = pt(v, r - 36)
        return (
          <g key={t}>
            <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="rgba(22,20,15,0.35)" />
            <text x={tx} y={ty} fill="rgba(22,20,15,0.45)" fontSize="9" textAnchor="middle" dominantBaseline="middle" fontFamily="JetBrains Mono">
              {['0', '10', '100', '1K', '10K', '100K', '1M', '10M'][t]}
            </text>
          </g>
        )
      })}
      {actual != null && (() => {
        const [x1, y1] = pt(actual, r + 10), [x2, y2] = pt(actual, r - 10)
        return <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="#16140f" strokeWidth="2.5" />
      })()}
      <g ref={needle}>
        <line x1={cx} y1={cy} x2={cx + r - 30} y2={cy} stroke="#16140f" strokeWidth="2" />
        <circle cx={cx + r - 30} cy={cy} r="3" fill="#0f6e5a" />
      </g>
      <circle cx={cx} cy={cy} r="7" fill="#efeae0" stroke="#0f6e5a" strokeWidth="2" />
    </svg>
  )
}

/** Model output across the what-if values, log y axis. */
function Sweep({ sweep, at, home }: { sweep: [number, number, number][]; at: number; home: number }) {
  const w = 320, h = 110, n = sweep.length
  const x = (i: number) => 8 + (i / (n - 1)) * (w - 16)
  const ymax = Math.max(...sweep.map((s) => lg(s[2])), 1)
  const y = (v: number) => h - 8 - (lg(v) / ymax) * (h - 20)
  const line = sweep.map((s, i) => `${i ? 'L' : 'M'}${x(i)} ${y(s[0])}`).join('')
  const band = sweep.map((s, i) => `${i ? 'L' : 'M'}${x(i)} ${y(s[2])}`).join('') +
    sweep.map((_, i) => `L${x(n - 1 - i)} ${y(sweep[n - 1 - i][1])}`).join('') + 'Z'
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full" aria-hidden>
      <path d={band} fill="#0f6e5a" fillOpacity="0.1" />
      <path d={line} fill="none" stroke="#0f6e5a" strokeWidth="1.6" />
      <line x1={x(home)} x2={x(home)} y1={6} y2={h - 6} stroke="rgba(22,20,15,0.3)" strokeDasharray="2 3" />
      <circle cx={x(at)} cy={y(sweep[at][0])} r="4.5" fill="#efeae0" stroke="#16140f" strokeWidth="2" style={{ transition: 'cx 0.3s, cy 0.3s' }} />
    </svg>
  )
}

export function Forecaster() {
  const [idx, setIdx] = useState(() => FLOWS.findIndex((f) => f.actual > 50_000 && f.actual < 100_000))
  const flow = FLOWS[Math.max(0, idx)]
  const [at, setAt] = useState(flow.whatIfAt)
  useEffect(() => setAt(flow.whatIfAt), [flow])

  const home = at === flow.whatIfAt
  const [p, lo, hi] = home ? [flow.pred, flow.lo, flow.hi] : flow.sweep[at]
  const err = flow.actual > 0 ? Math.abs(flow.pred - flow.actual) / flow.actual : null
  const best = R.models[0]
  const fmt = useMemo(() => (v: number) => bytes(v), [])

  return (
    <Chapter
      id="forecaster"
      index="09"
      eyebrow={`Regression · ${D.overview.dataset.name} · ${R.best}`}
      title={<>How much<br />comes back?</>}
      deck={<>Before a response finishes, can we say how big it will be? A {R.best} of 200 trees predicts the bytes a flow returns from {R.features} other statistics. On {int(D.classification.testRows)} unseen flows: R² {best.r2.toFixed(4)}, typical error {int(best.mae)} bytes.</>}
    >
      <div className="mb-6 flex gap-2 overflow-x-auto pb-2 [scrollbar-width:thin]" role="group" aria-label="Choose a flow">
        {FLOWS.map((f, i) => (
          <button key={f.id} type="button" onClick={() => setIdx(i)}
            className={cn('grid shrink-0 gap-1 rounded-xl border px-4 py-3 text-left transition',
              i === idx ? 'border-signal bg-signal/[0.06]' : 'border-hairline-strong hover:border-ink-dim')}>
            <span className="num text-sm text-ink">{bytes(f.actual)}</span>
            <span className="flex items-center gap-1.5 text-[0.7rem] text-ink-muted">
              <span className="h-1.5 w-1.5 rounded-full" style={{ background: CLASS_COLOR[f.label] }} />
              {f.f.proto} · {f.f.service === '-' ? 'no service' : f.f.service}
            </span>
          </button>
        ))}
      </div>

      <div className="grid gap-px overflow-hidden rounded-2xl border border-hairline-strong bg-hairline-strong lg:grid-cols-[minmax(0,7fr)_minmax(0,5fr)]">
        <div className="grid content-start justify-items-center gap-6 bg-panel p-6 md:p-10">
          <Gauge pred={p} lo={lo} hi={hi} actual={home ? flow.actual : null} />
          <div className="grid w-full gap-6 sm:grid-cols-2">
            <div className="grid gap-1">
              <span className="eyebrow">{home ? 'Predicted' : 'What-if prediction'}</span>
              <Counter value={p} format={fmt} className="num text-[clamp(2.4rem,5vw,3.6rem)] font-medium leading-none text-signal" />
              <span className="num text-xs text-ink-dim">trees agree on {bytes(lo)} to {bytes(hi)} (10th to 90th percentile)</span>
            </div>
            <div className="grid gap-1 sm:text-right">
              <span className="eyebrow">Actually sent</span>
              <span className={cn('num text-[clamp(2.4rem,5vw,3.6rem)] font-medium leading-none transition-opacity', home ? 'text-ink' : 'text-ink-dim')}>{bytes(flow.actual)}</span>
              <span className="num text-xs text-ink-dim">
                {home ? (err == null ? (flow.pred === 0 ? 'predicted exactly: nothing came back' : 'nothing came back') : `off by ${pct(err, 1)}`) : 'only known for the real packet count'}
              </span>
            </div>
          </div>
        </div>

        <div className="grid content-start gap-7 bg-panel p-6 md:p-8">
          <div className="grid gap-3">
            <div className="flex items-baseline justify-between gap-4">
              <label htmlFor="whatif" className="eyebrow">What if it returned</label>
              <output htmlFor="whatif" className="num text-xl font-medium">{int(W.values[at])} packets</output>
            </div>
            <Slider id="whatif" value={[at]} min={0} max={W.values.length - 1} step={1} onValueChange={([v]) => setAt(v)} aria-label="Packets returned" />
            <p className="text-[0.8rem] leading-snug text-ink-dim">
              Changes only the packet count, holding the flow's other {R.features - 1} features fixed. The dashed line is the real value ({int(flow.f.dpkts)}).
            </p>
          </div>
          <Sweep sweep={flow.sweep} at={at} home={flow.whatIfAt} />
          <dl className="grid grid-cols-2 gap-px overflow-hidden rounded-xl bg-hairline-strong text-sm">
            {[
              ['True class', flow.label],
              ['Duration', `${flow.f.dur.toFixed(3)} s`],
              ['Packets sent', int(flow.f.spkts)],
              ['Bytes sent', bytes(flow.f.sbytes)],
              ['State', flow.f.state],
              ['TTL', `${flow.f.sttl} / ${flow.f.dttl}`],
            ].map(([k, v]) => (
              <div key={k} className="grid gap-0.5 bg-panel-2 p-3">
                <dt className="text-[0.7rem] text-ink-dim">{k}</dt>
                <dd className="num">{v}</dd>
              </div>
            ))}
          </dl>
          <p className="text-xs leading-relaxed text-ink-dim">
            Packets returned carry {pct(R.importance[0].value)} of the forest's split importance, so it moves the forecast most. The rise is not a straight line: the trees learned different byte sizes per packet for different kinds of traffic.
          </p>
        </div>
      </div>
    </Chapter>
  )
}
