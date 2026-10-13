import { useMemo, useRef, useState } from 'react'
import { Chapter } from '@/components/chapter'
import { gsap, useGSAP, prefersReducedMotion } from '@/lib/gsap'
import { T, int, type TonModel } from '@/lib/data'
import { cn } from '@/lib/utils'

const C = T.classification

type Metric = 'acc' | 'precision' | 'recall' | 'f1w' | 'f1m' | 'auc' | 'fit'
const METRICS: { key: Metric; label: string; hint: string }[] = [
  { key: 'f1w', label: 'Weighted F1', hint: 'F1 per class, averaged by class size. The headline metric.' },
  { key: 'f1m', label: 'Macro F1', hint: 'Every class counts equally, so the rare MITM class weighs as much as normal traffic.' },
  { key: 'acc', label: 'Accuracy', hint: 'Share of the test flows labelled correctly.' },
  { key: 'precision', label: 'Precision', hint: 'Of the flows given a class, how many really were that class (weighted).' },
  { key: 'recall', label: 'Recall', hint: 'Of the flows in a class, how many were found (weighted).' },
  { key: 'auc', label: 'ROC-AUC', hint: 'One-vs-rest, macro-averaged: how well the probabilities rank each class above the rest.' },
  { key: 'fit', label: 'Training time', hint: 'Seconds to train on the full training set. Shorter is better; SVC was trained on a 20,000-row sample.' },
]

const ROW = 46

/** Bars that race out on scroll and re-rank themselves when the metric changes. */
function Race() {
  const root = useRef<HTMLDivElement>(null)
  const [metric, setMetric] = useState<Metric>('f1w')
  const [zoom, setZoom] = useState(true)
  const m = METRICS.find((x) => x.key === metric)!
  const lowerBetter = metric === 'fit'
  const ranked = useMemo(() => [...C.models].sort((a, b) => (lowerBetter ? a[metric] - b[metric] : b[metric] - a[metric])), [metric, lowerBetter])
  const rank = (name: string) => ranked.findIndex((r) => r.model === name)

  const values = C.models.map((r) => r[metric])
  const lo = lowerBetter ? 0 : zoom ? Math.max(0, Math.floor((Math.min(...values.filter((v) => v > 0.6)) - 0.02) * 50) / 50) : 0
  const hi = lowerBetter ? Math.max(...values) : 1
  const width = (v: number) => `${Math.max(0.8, ((Math.max(lo, v) - lo) / (hi - lo)) * 100)}%`

  useGSAP(() => {
    if (prefersReducedMotion()) return
    gsap.from(root.current!.querySelectorAll('[data-fill]'), {
      scaleX: 0, duration: 1.4, ease: 'expo.out', stagger: { each: 0.08, from: 'end' },
      scrollTrigger: { trigger: root.current, start: 'top 75%', once: true },
    })
  }, { scope: root })

  return (
    <div ref={root} className="grain grid content-start gap-6 rounded-2xl border border-hairline-strong bg-panel p-5 md:p-8">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap gap-1 rounded-full border border-hairline-strong bg-paper p-1" role="tablist">
          {METRICS.map((x) => (
            <button key={x.key} type="button" role="tab" aria-selected={metric === x.key} onClick={() => setMetric(x.key)}
              className={cn('rounded-full px-3 py-1.5 text-xs font-medium transition', metric === x.key ? 'bg-ink text-paper' : 'text-ink-muted hover:text-ink')}>
              {x.label}
            </button>
          ))}
        </div>
        {!lowerBetter && (
          <button type="button" onClick={() => setZoom((z) => !z)} className="chip transition hover:border-ink">
            {zoom ? `zoomed: axis starts at ${lo.toFixed(2)}` : 'full scale 0 to 1'} · switch
          </button>
        )}
      </div>

      <div className="relative" style={{ height: ROW * ranked.length }}>
        {C.models.map((r) => {
          const i = rank(r.model)
          const final = r.model === C.best
          return (
            <div key={r.model} className="absolute inset-x-0 grid grid-cols-[1.6rem_9.5rem_minmax(0,1fr)_4.6rem] items-center gap-3 transition-transform duration-700 ease-[cubic-bezier(.16,1,.3,1)]"
              style={{ height: ROW, transform: `translateY(${i * ROW}px)` }}>
              <span className="num text-sm text-ink-dim">{String(i + 1).padStart(2, '0')}</span>
              <span className="flex items-center gap-2 text-sm">
                <span className={cn('num rounded-sm px-1 text-[0.6rem]', r.part === 'A' ? 'bg-panel-2 text-ink-muted' : 'bg-ink text-paper')}>{r.part}</span>
                <span className={final ? 'font-semibold' : 'text-ink-muted'}>{r.model}</span>
              </span>
              <span className="relative h-5 overflow-hidden rounded-sm bg-panel-2">
                <span data-fill className={cn('absolute inset-y-0 left-0 origin-left transition-[width] duration-700', final ? 'bg-red' : r.part === 'A' ? 'bg-ink/45' : 'bg-ink/80')}
                  style={{ width: width(r[metric]) }} />
              </span>
              <span className="num text-right text-sm">{metric === 'fit' ? `${r.fit.toFixed(r.fit < 10 ? 1 : 0)} s` : r[metric].toFixed(4)}</span>
            </div>
          )
        })}
      </div>
      <div className="flex flex-wrap items-center justify-between gap-3 border-t border-hairline-strong pt-4 text-sm text-ink-muted">
        <p className="max-w-[60ch]">{m.hint}</p>
        <span className="flex items-center gap-3 text-xs">
          <span className="flex items-center gap-1.5"><span className="h-2.5 w-4 rounded-sm bg-ink/45" />Part A</span>
          <span className="flex items-center gap-1.5"><span className="h-2.5 w-4 rounded-sm bg-ink/80" />Part B</span>
          <span className="flex items-center gap-1.5"><span className="h-2.5 w-4 rounded-sm bg-red" />final model</span>
        </span>
      </div>
    </div>
  )
}

/** "{'learning_rate': 0.05, 'max_iter': 200}" -> [["learning_rate", "0.05"], ["max_iter", "200"]] */
function parseParams(s: string): [string, string][] {
  if (!s.startsWith('{')) return []
  return [...s.matchAll(/'([^']+)':\s*('([^']*)'|\([^)]*\)|[^,}]+)/g)].map((m) => [m[1], (m[3] ?? m[2]).trim()])
}

function ModelCard({ r, rankNo }: { r: TonModel; rankNo: number }) {
  const final = r.model === C.best
  const params = parseParams(r.params)
  return (
    <article className={cn('grain relative grid content-start gap-4 rounded-xl border p-5 transition hover:-translate-y-1 hover:shadow-[0_16px_36px_rgba(22,20,15,0.12)]',
      final ? 'border-red bg-panel' : 'border-hairline-strong bg-panel')}>
      {final && <span className="stamp absolute -top-3 right-4 bg-paper text-red">Final model</span>}
      <div className="flex items-start justify-between gap-3">
        <div className="grid gap-0.5">
          <span className="eyebrow">#{rankNo} · Part {r.part}</span>
          <h3 className="text-[1.05rem] font-semibold leading-tight">{r.model}</h3>
        </div>
        <span className="num text-[1.4rem] font-medium leading-none">{r.f1w.toFixed(3)}</span>
      </div>
      <dl className="num grid gap-1 border-t border-dashed border-hairline-strong pt-3 text-xs">
        {params.length ? params.map(([k, v]) => (
          <div key={k} className="flex justify-between gap-3"><dt className="text-ink-dim">{k}</dt><dd>{v}</dd></div>
        )) : <div className="text-ink-dim">no hyperparameters to tune</div>}
      </dl>
      <div className="flex justify-between gap-3 text-[0.7rem] text-ink-dim">
        <span>{r.tried ? `${r.tried} settings tried, CV F1 ${r.cvTune?.toFixed(3)}` : 'library defaults'}</span>
        <span>{r.fit < 10 ? r.fit.toFixed(1) : int(r.fit)} s</span>
      </div>
    </article>
  )
}

/** Library defaults -> tuned on a sample -> final full search, for the chosen model. */
function TuningSteps() {
  const s = C.final.stages
  const base = s[0].f1w
  const max = Math.max(...s.map((x) => x.f1w))
  return (
    <div className="grain grid content-start gap-5 rounded-2xl border border-hairline-strong bg-panel p-6 md:p-8">
      <span className="eyebrow">Tuning the final model · {C.best}</span>
      <ol className="grid gap-4">
        {s.map((x, i) => (
          <li key={x.stage} className="grid grid-cols-[2rem_minmax(0,1fr)_auto] items-center gap-3">
            <span className={cn('num grid h-8 w-8 place-items-center rounded-full border text-sm', x.f1w === max ? 'border-red bg-red text-paper' : 'border-hairline-strong')}>{i + 1}</span>
            <span className="grid gap-0.5">
              <span className="text-sm">{x.stage.replace(/^\d\.\s*/, '')}</span>
              <span className="num text-xs text-ink-dim">macro F1 {x.f1m.toFixed(4)} · ROC-AUC {x.auc.toFixed(4)}</span>
            </span>
            <span className="grid justify-items-end">
              <span className="num text-lg font-medium">{x.f1w.toFixed(4)}</span>
              <span className={cn('num text-xs', x.f1w > base ? 'text-signal' : 'text-ink-dim')}>{i === 0 ? 'baseline' : `${x.f1w >= base ? '+' : ''}${(x.f1w - base).toFixed(4)}`}</span>
            </span>
          </li>
        ))}
      </ol>
      <p className="text-xs leading-relaxed text-ink-dim">Weighted F1 on the test set. Settings were always chosen by cross-validation on training data; the test set only reports the result.</p>
    </div>
  )
}

/** Five fold scores per model on one shared axis. */
function Folds() {
  const all = C.cv.flatMap((c) => c.folds)
  const lo = Math.min(...all) - 0.001, hi = Math.max(...all) + 0.001
  const x = (v: number) => `${((v - lo) / (hi - lo)) * 100}%`
  return (
    <div className="grain grid content-start gap-5 rounded-2xl border border-hairline-strong bg-panel p-6 md:p-8">
      <span className="eyebrow">5-fold cross-validation · top two</span>
      {C.cv.map((c, k) => (
        <div key={c.model} className="grid gap-2">
          <div className="flex items-baseline justify-between gap-3 text-sm">
            <span className={k === 0 ? 'font-semibold' : 'text-ink-muted'}>{c.model}</span>
            <span className="num">{c.mean.toFixed(4)} <span className="text-ink-dim">± {c.std.toFixed(4)}</span></span>
          </div>
          <div className="relative h-7 rounded-sm bg-panel-2">
            <span className="absolute inset-y-1 w-px bg-ink/40" style={{ left: x(c.mean) }} />
            {c.folds.map((f, i) => (
              <span key={i} title={`fold ${i + 1}: ${f}`} className={cn('absolute top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-paper', k === 0 ? 'bg-red' : 'bg-ink')}
                style={{ left: x(f) }} />
            ))}
          </div>
        </div>
      ))}
      <div className="num flex justify-between text-[0.65rem] text-ink-dim"><span>{lo.toFixed(4)}</span><span>weighted F1 per fold</span><span>{hi.toFixed(4)}</span></div>
      <p className="text-xs leading-relaxed text-ink-dim">Each dot is one fold of the training set. Tight clusters mean the ranking does not depend on a lucky split.</p>
    </div>
  )
}

export function Leaderboard() {
  return (
    <Chapter
      id="models"
      index="04"
      eyebrow="Ten classifiers, one test set"
      title={<>Ten models,<br />one shared exam</>}
      deck={<>Five Part A baselines and five Part B ensembles and networks, all trained on the same {int(C.trainRows)} flows and scored on the same {int(C.testRows)}. Every model's settings were tuned by cross-validation first. Switch the metric and watch the order change.</>}
    >
      <Race />
      <div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
        {C.models.map((r, i) => <ModelCard key={r.model} r={r} rankNo={i + 1} />)}
      </div>
      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <TuningSteps />
        <Folds />
      </div>
    </Chapter>
  )
}
