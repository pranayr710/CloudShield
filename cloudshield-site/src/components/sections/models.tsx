import { useMemo, useRef, useState } from 'react'
import { Chapter } from '@/components/chapter'
import { gsap, useGSAP, prefersReducedMotion } from '@/lib/gsap'
import { D, int, pct } from '@/lib/data'
import { cn } from '@/lib/utils'

const R = D.regression

type View = 'r2' | 'rmse'

/** Validation vs test R² per model on one axis: the gap is overfitting. */
function Dumbbell() {
  const root = useRef<HTMLDivElement>(null)
  const [view, setView] = useState<View>('r2')
  const rows = useMemo(() => [...R.models].sort((a, b) => (view === 'r2' ? b.r2 - a.r2 : a.rmse - b.rmse)), [view])
  const lo = 0.8
  const x = (v: number) => `${((Math.max(lo, v) - lo) / (1 - lo)) * 100}%`
  const maxRmse = Math.max(...R.models.map((m) => m.rmse), R.baselineRmse)

  useGSAP(() => {
    if (prefersReducedMotion()) return
    gsap.from(root.current!.querySelectorAll('[data-row]'), {
      opacity: 0, x: -30, stagger: 0.05, duration: 0.8, ease: 'expo.out',
      scrollTrigger: { trigger: root.current, start: 'top 80%', once: true },
    })
  }, { scope: root })

  return (
    <div ref={root} className="grid content-start gap-6 rounded-2xl border border-hairline-strong bg-panel p-6 md:p-8">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <span className="eyebrow">Ten regressors</span>
        <div className="flex gap-1 rounded-full border border-hairline-strong p-1" role="tablist">
          {([['r2', 'R², validation vs test'], ['rmse', 'Error in bytes']] as const).map(([k, l]) => (
            <button key={k} type="button" role="tab" aria-selected={view === k} onClick={() => setView(k)}
              className={cn('rounded-full px-3 py-1.5 text-xs font-medium transition', view === k ? 'bg-signal text-paper' : 'text-ink-muted hover:text-ink')}>{l}</button>
          ))}
        </div>
      </div>
      <ol className="grid gap-3.5">
        {rows.map((m, i) => {
          const gap = m.r2Val - m.r2
          const overfit = gap > 0.01
          return (
            <li key={m.model} data-row className="grid grid-cols-[1.5rem_8.5rem_minmax(0,1fr)_4.5rem] items-center gap-3 text-sm">
              <span className="num text-ink-dim">{i + 1}</span>
              <span className={cn(i === 0 ? 'text-ink' : 'text-ink-muted', overfit && view === 'r2' && 'text-red')}>{m.model}</span>
              {view === 'r2' ? (
                <span className="relative h-4">
                  <span className="absolute inset-x-0 top-1/2 h-px bg-hairline-strong" />
                  <span className="absolute top-1/2 h-[3px] -translate-y-1/2 rounded-full bg-red/70 transition-all duration-700"
                    style={{ left: x(Math.min(m.r2, m.r2Val)), width: `calc(${x(Math.max(m.r2, m.r2Val))} - ${x(Math.min(m.r2, m.r2Val))})` }} />
                  <span className="absolute top-1/2 h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full border border-ink-muted bg-paper transition-all duration-700" style={{ left: x(m.r2Val) }} title="validation" />
                  <span className={cn('absolute top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full transition-all duration-700', i === 0 ? 'bg-signal' : 'bg-ink')} style={{ left: x(m.r2) }} title="test" />
                </span>
              ) : (
                <span className="relative h-3 rounded-full bg-panel-2">
                  <span className={cn('absolute inset-y-0 left-0 rounded-full transition-[width] duration-700', i === 0 ? 'bg-signal' : 'bg-ink/70')} style={{ width: `${(m.rmse / maxRmse) * 100}%` }} />
                  <span className="absolute inset-y-[-5px] w-px bg-red" style={{ left: `${(R.baselineRmse / maxRmse) * 100}%` }} />
                </span>
              )}
              <span className="num text-right">{view === 'r2' ? m.r2.toFixed(4) : `${int(m.rmse / 1000)}K`}</span>
            </li>
          )
        })}
      </ol>
      <p className="text-sm leading-relaxed text-ink-muted">
        {view === 'r2'
          ? <>Hollow dot: validation R². Solid dot: the untouched test set. Axis starts at 0.80. A long red bar is a model that looked better in training than it is, which is what happened to the degree-2 polynomial.</>
          : <>Root-mean-square error converted back to bytes. The red line is a model that always predicts the average ({int(R.baselineRmse / 1000)}K). The linear models beat it on R² but not on the biggest transfers.</>}
      </p>
    </div>
  )
}

/** Predicted vs actual on log axes; a perfect model draws the diagonal. */
function Scatter() {
  const max = Math.max(...R.scatter.flatMap((p) => p))
  const s = (v: number) => (v / max) * 280 + 10
  return (
    <div className="grid content-start gap-4 rounded-2xl border border-hairline-strong bg-panel p-6 md:p-8">
      <div className="flex items-baseline justify-between gap-4">
        <span className="eyebrow">Predicted vs actual</span>
        <span className="text-xs text-ink-dim">{int(R.scatter.length)} test flows, log scale</span>
      </div>
      <svg viewBox="0 0 300 300" className="w-full" role="img" aria-label="Scatter plot of predicted against actual bytes; points lie along the diagonal">
        {[0, 2, 4, 6, 8, 10, 12, 14].filter((t) => t <= max).map((t) => (
          <g key={t}>
            <line x1={s(t)} x2={s(t)} y1={10} y2={290} stroke="rgba(22,20,15,0.06)" />
            <line y1={300 - s(t)} y2={300 - s(t)} x1={10} x2={290} stroke="rgba(22,20,15,0.06)" />
          </g>
        ))}
        <line x1={10} y1={290} x2={290} y2={10} stroke="#e5481c" strokeOpacity="0.6" strokeDasharray="4 4" />
        {R.scatter.map(([a, p], i) => (
          <circle key={i} cx={s(a)} cy={300 - s(p)} r="1.6" fill="#0f6e5a" fillOpacity="0.45" />
        ))}
      </svg>
      <p className="text-sm leading-relaxed text-ink-muted">
        Each dot is one flow. The dots at the bottom left are flows that sent nothing back, and the forest predicts them as nothing too.
      </p>
    </div>
  )
}

function Importance() {
  const root = useRef<HTMLDivElement>(null)
  const top = R.importance.slice(0, 8)
  useGSAP(() => {
    if (prefersReducedMotion()) return
    gsap.from(root.current!.querySelectorAll('[data-bar]'), {
      scaleX: 0, duration: 1.2, stagger: 0.06, ease: 'expo.out',
      scrollTrigger: { trigger: root.current, start: 'top 80%', once: true },
    })
  }, { scope: root })
  return (
    <div ref={root} className="grid content-start gap-4 rounded-2xl border border-hairline-strong bg-panel p-6 md:p-8">
      <span className="eyebrow">What the forest splits on</span>
      {top.map((f) => (
        <div key={f.name} className="grid grid-cols-[7rem_minmax(0,1fr)_3.5rem] items-center gap-3 text-sm">
          <span className="num truncate text-ink-muted">{f.name}</span>
          <span className="relative h-2.5 rounded-full bg-panel-2">
            <span data-bar className="absolute inset-y-0 left-0 origin-left rounded-full bg-signal" style={{ width: `${Math.max(0.8, Math.sqrt(f.value / top[0].value) * 100)}%` }} />
          </span>
          <span className="num text-right">{pct(f.value, 1)}</span>
        </div>
      ))}
      <p className="text-sm leading-relaxed text-ink-muted">
        Square-root scale. Packets returned and packets lost on the way back do most of the work; nothing else gets past one percent.
      </p>
    </div>
  )
}

function Tuning() {
  return (
    <div className="grain grid border border-hairline-strong content-start gap-6 rounded-2xl bg-panel p-6 md:p-8">
      <span className="eyebrow">Tuned with GridSearchCV, then cross-validated</span>
      <div className="grid gap-4 sm:grid-cols-2">
        {R.tuning.map((t) => (
          <div key={t.model} className="grid gap-2 rounded-xl border border-hairline-strong p-4">
            <span className="text-sm text-ink">{t.model}</span>
            <div className="flex items-baseline gap-2">
              <span className="num text-ink-dim line-through">{int(t.rmseBefore)}</span>
              <span className="text-ink-dim">→</span>
              <span className="num text-2xl font-medium text-signal">{int(t.rmseAfter)}</span>
              <span className="text-xs text-ink-dim">bytes RMSE</span>
            </div>
            <span className="num text-xs text-ink-muted">R² {t.r2Before.toFixed(4)} → {t.r2After.toFixed(4)} · {pct(1 - t.rmseAfter / t.rmseBefore, 0)} less error</span>
          </div>
        ))}
      </div>
      <div className="grid gap-2 text-sm">
        <span className="eyebrow">5-fold CV R² on the training set</span>
        {R.cv.map((c) => (
          <div key={c.model} className="grid grid-cols-[minmax(0,1fr)_auto] gap-4 border-b border-hairline py-1.5">
            <span className="text-ink-muted">{c.model}</span><span className="num">{c.mean.toFixed(4)} ± {c.std.toFixed(4)}</span>
          </div>
        ))}
      </div>
      <p className="text-sm leading-relaxed text-ink-muted">Small spreads across folds mean the ranking is stable, not a lucky split.</p>
    </div>
  )
}

export function Models() {
  return (
    <Chapter
      id="regression-models"
      index="09"
      eyebrow={`Regression · ${D.overview.dataset.name} · model comparison`}
      title={<>Ten regressors,<br />one winner</>}
      deck={<>Linear models, regularised ones, kernels, neighbours and trees, all trained on the same {int(R.trainRows)} flows and scored on the same test set. Tree ensembles win because traffic volume is full of thresholds and interactions a straight line cannot draw.</>}
    >
      <div className="grid gap-4 lg:grid-cols-[minmax(0,7fr)_minmax(0,5fr)]">
        <Dumbbell />
        <Scatter />
        <Importance />
        <Tuning />
      </div>
    </Chapter>
  )
}
