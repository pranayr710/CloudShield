import { useMemo, useRef, useState } from 'react'
import { Chapter } from '@/components/chapter'
import { gsap, useGSAP, prefersReducedMotion } from '@/lib/gsap'
import { T, int, type TonModel } from '@/lib/data'
import { cn } from '@/lib/utils'

const C = T.classification

type Metric = 'f1w' | 'f1m' | 'acc' | 'precision' | 'recall' | 'auc'
const METRICS: { key: Metric; label: string; hint: string }[] = [
  { key: 'f1w', label: 'Weighted F1', hint: 'F1 per class, averaged by class size. The headline metric.' },
  { key: 'f1m', label: 'Macro F1', hint: 'Every class counts equally, so the weak DoS class pulls it down.' },
  { key: 'acc', label: 'Accuracy', hint: 'Share of the test flows labelled correctly.' },
  { key: 'precision', label: 'Precision', hint: 'Of the flows given a class, how many really were that class (weighted by class size).' },
  { key: 'recall', label: 'Recall', hint: 'Of the flows in a class, how many were found (weighted by class size).' },
  { key: 'auc', label: 'ROC-AUC', hint: 'One-vs-rest, averaged over the classes in the test fold: how well the scores rank each class above the rest.' },
]
const ROW = 42
const val = (r: TonModel, k: Metric) => r[k] ?? 0
const HAS = (k: Metric) => C.models.every((m) => m[k] != null)

/** Bars that race out on scroll and re-rank themselves when the metric changes. */
function Race() {
  const root = useRef<HTMLDivElement>(null)
  const [metric, setMetric] = useState<Metric>('f1w')
  const m = METRICS.find((x) => x.key === metric)!
  const ranked = useMemo(() => [...C.models].sort((a, b) => val(b, metric) - val(a, metric)), [metric])
  const rank = (key: string) => ranked.findIndex((r) => r.key === key)

  useGSAP(() => {
    if (prefersReducedMotion()) return
    gsap.from(root.current!.querySelectorAll('[data-fill]'), {
      scaleX: 0, duration: 1.4, ease: 'expo.out', stagger: { each: 0.07, from: 'end' },
      scrollTrigger: { trigger: root.current, start: 'top 75%', once: true },
    })
  }, { scope: root })

  return (
    <div ref={root} className="grain grid content-start gap-6 rounded-2xl border border-hairline-strong bg-panel p-5 md:p-8">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap gap-1 rounded-full border border-hairline-strong bg-paper p-1" role="tablist">
          {METRICS.filter((x) => HAS(x.key)).map((x) => (
            <button key={x.key} type="button" role="tab" aria-selected={metric === x.key} onClick={() => setMetric(x.key)}
              className={cn('rounded-full px-3 py-1.5 text-xs font-medium transition', metric === x.key ? 'bg-ink text-paper' : 'text-ink-muted hover:text-ink')}>
              {x.label}
            </button>
          ))}
        </div>
        <span className="chip">test set: {int(C.testRows)} flows from {C.testMachines} unseen machines</span>
      </div>

      <div className="relative" style={{ height: ROW * ranked.length }}>
        {C.models.map((r) => {
          const i = rank(r.key)
          const final = r.key === C.best
          return (
            <div key={r.key} className="absolute inset-x-0 grid grid-cols-[1.6rem_11rem_minmax(0,1fr)_4.2rem] items-center gap-3 transition-transform duration-700 ease-[cubic-bezier(.16,1,.3,1)]"
              style={{ height: ROW, transform: `translateY(${i * ROW}px)` }}>
              <span className="num text-sm text-ink-dim">{String(i + 1).padStart(2, '0')}</span>
              <span className="flex items-center gap-2 text-sm">
                <span className={cn('num rounded-sm px-1 text-[0.6rem]', r.part === 'A' ? 'bg-panel-2 text-ink-muted' : 'bg-ink text-paper')}>{r.part}</span>
                <span className={cn('truncate', final ? 'font-semibold' : 'text-ink-muted')}>{r.model}</span>
              </span>
              <span className="relative h-5 overflow-hidden rounded-sm bg-panel-2">
                <span data-fill className={cn('absolute inset-y-0 left-0 origin-left transition-[width] duration-700', final ? 'bg-red' : r.tuned ? 'bg-amber' : r.part === 'A' ? 'bg-ink/45' : 'bg-ink/80')}
                  style={{ width: `${Math.max(1, val(r, metric) * 100)}%` }} />
              </span>
              <span className="num text-right text-sm">{val(r, metric).toFixed(4)}</span>
            </div>
          )
        })}
      </div>
      <div className="flex flex-wrap items-center justify-between gap-3 border-t border-hairline-strong pt-4 text-sm text-ink-muted">
        <p className="max-w-[60ch]">{m.hint}</p>
        <span className="flex flex-wrap items-center gap-3 text-xs">
          <span className="flex items-center gap-1.5"><span className="h-2.5 w-4 rounded-sm bg-ink/45" />Part A</span>
          <span className="flex items-center gap-1.5"><span className="h-2.5 w-4 rounded-sm bg-ink/80" />Part B</span>
          <span className="flex items-center gap-1.5"><span className="h-2.5 w-4 rounded-sm bg-amber" />tuned</span>
          <span className="flex items-center gap-1.5"><span className="h-2.5 w-4 rounded-sm bg-red" />final model</span>
        </span>
      </div>
    </div>
  )
}

function ModelCard({ r, rankNo }: { r: TonModel; rankNo: number }) {
  const final = r.key === C.best
  return (
    <article className={cn('grain relative grid content-start gap-3 rounded-xl border bg-panel p-5 transition hover:-translate-y-1 hover:shadow-[0_16px_36px_rgba(22,20,15,0.12)]',
      final ? 'border-red' : 'border-hairline-strong')}>
      {final && <span className="stamp absolute -top-3 right-4 bg-paper text-red">Final model</span>}
      <div className="flex items-start justify-between gap-3">
        <div className="grid gap-0.5">
          <span className="eyebrow">#{rankNo} · Part {r.part}{r.tuned ? ' · tuned' : ''}</span>
          <h3 className="text-[1.02rem] font-semibold leading-tight">{r.model}</h3>
        </div>
        <span className="num text-[1.35rem] font-medium leading-none">{r.f1w.toFixed(3)}</span>
      </div>
      <p className="num border-t border-dashed border-hairline-strong pt-3 text-xs text-ink-muted">{r.params}</p>
      <div className="num flex justify-between gap-3 text-[0.7rem] text-ink-dim">
        <span>macro F1 {r.f1m.toFixed(3)}</span><span>accuracy {r.acc.toFixed(3)}</span>{r.auc != null && <span>AUC {r.auc.toFixed(3)}</span>}
      </div>
    </article>
  )
}

/** Baseline -> tuned, for the two models the notebook tuned. */
function TuningSteps() {
  const pairs = [['GradientBoostingClassifier', 'GBC (tuned)'], ['Bagging', 'Bagging (tuned)']] as const
  return (
    <div className="grain grid content-start gap-5 rounded-2xl border border-hairline-strong bg-panel p-6 md:p-8">
      <span className="eyebrow">Hyperparameter tuning · RandomizedSearchCV</span>
      {pairs.map(([a, b]) => {
        const base = C.models.find((m) => m.key === a)!, tuned = C.models.find((m) => m.key === b)!
        const gain = tuned.f1w - base.f1w
        return (
          <div key={a} className="grid gap-3 rounded-xl border border-hairline-strong bg-paper p-4">
            <span className="text-sm font-semibold">{base.model}</span>
            <div className="grid grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)] items-center gap-3">
              <div className="grid gap-0.5"><span className="eyebrow">baseline</span><span className="num text-lg">{base.f1w.toFixed(4)}</span><span className="num text-[0.65rem] text-ink-dim">{base.params}</span></div>
              <span className="text-ink-dim">→</span>
              <div className="grid gap-0.5"><span className="eyebrow">tuned</span><span className={cn('num text-lg', gain > 0 ? 'text-signal' : 'text-red')}>{tuned.f1w.toFixed(4)}</span><span className="num text-[0.65rem] text-ink-dim">{tuned.params}</span></div>
            </div>
            <span className={cn('num text-xs', gain > 0 ? 'text-signal' : 'text-red')}>{gain > 0 ? '+' : ''}{gain.toFixed(4)} weighted F1 on the unseen machines</span>
          </div>
        )
      })}
    </div>
  )
}

/** The search's cross-validation score next to the score on unseen machines. */
function SearchGap() {
  return (
    <div className="screen grain grid content-start gap-5 rounded-2xl p-6 md:p-8">
      <span className="eyebrow text-red">Cross-validation vs unseen machines</span>
      {C.searches.map((s) => (
        <div key={s.model} className="grid gap-2">
          <div className="flex items-baseline justify-between gap-3 text-sm">
            <span className="font-semibold">{s.model}</span>
            <span className="num text-ink-muted">gap {(s.cv - s.test).toFixed(3)}</span>
          </div>
          {[['CV on training machines', s.cv, 'bg-phosphor'], ['test on unseen machines', s.test, 'bg-red']].map(([k, v, col]) => (
            <div key={k as string} className="grid grid-cols-[10rem_minmax(0,1fr)_3.6rem] items-center gap-3 text-xs">
              <span className="text-ink-muted">{k}</span>
              <span className="relative h-3 rounded-sm bg-panel-2"><span className={cn('absolute inset-y-0 left-0 rounded-sm', col as string)} style={{ width: `${(v as number) * 100}%` }} /></span>
              <span className="num text-right">{(v as number).toFixed(4)}</span>
            </div>
          ))}
        </div>
      ))}
      <p className="text-xs leading-relaxed text-ink-dim">
        The search scored each setting by 3-fold cross-validation on a 35% sample of the training rows. Those folds share machines, so the scores look near-perfect. The test fold holds machines the model never saw; the drop is the honest measure of how it would do on a new network.
      </p>
    </div>
  )
}

/** 5-fold CV grouped by source machine, training set only. */
function Folds() {
  if (!C.cv5.length) return null
  const all = C.cv5.flatMap((c) => c.folds)
  const lo = Math.max(0, Math.min(...all) - 0.02), hi = Math.min(1, Math.max(...all) + 0.02)
  const x = (v: number) => `${((v - lo) / (hi - lo)) * 100}%`
  return (
    <div className="grain grid content-start gap-5 rounded-2xl border border-hairline-strong bg-panel p-6 md:p-8 lg:col-span-2">
      <div className="flex flex-wrap items-baseline justify-between gap-3">
        <span className="eyebrow">5-fold cross-validation of the top two · training set only</span>
        <span className="text-xs text-ink-dim">grouped: each fold validates on machines its training folds never saw · stratified: folds share machines</span>
      </div>
      {C.cv5.map((c, k) => (
        <div key={c.model} className="grid gap-2">
          <div className="flex flex-wrap items-baseline justify-between gap-3 text-sm">
            <span className={k === 0 ? 'font-semibold' : 'text-ink-muted'}>{c.model} <span className="chip ml-1">{c.kind}</span></span>
            <span className="num">weighted F1 {c.mean.toFixed(4)} <span className="text-ink-dim">± {c.std.toFixed(4)} · macro F1 {c.f1m.toFixed(3)} · accuracy {c.acc.toFixed(3)}</span></span>
          </div>
          <div className="relative h-8 rounded-sm bg-panel-2">
            <span className="absolute inset-y-1 w-px bg-ink/40" style={{ left: x(c.mean) }} />
            {c.folds.map((f, i) => (
              <span key={i} title={`fold ${i + 1}: ${f}`} className={cn('num absolute top-1/2 grid h-5 w-5 -translate-x-1/2 -translate-y-1/2 place-items-center rounded-full border-2 border-paper text-[0.55rem] text-paper', c.kind === 'grouped' ? 'bg-red' : 'bg-signal')}
                style={{ left: x(f) }}>{i + 1}</span>
            ))}
          </div>
        </div>
      ))}
      <div className="num flex justify-between text-[0.65rem] text-ink-dim"><span>{lo.toFixed(2)}</span><span>weighted F1 per fold</span><span>{hi.toFixed(2)}</span></div>
      <p className="text-xs leading-relaxed text-ink-dim">Each numbered dot is one fold. Grouped folds that hold every machine of an attack type never train on it, and their score collapses; stratified folds share machines, so they stay high and hide that risk.</p>
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
      deck={<>Five Part A baselines and five Part B ensembles and networks, plus the two that were tuned, all trained on {int(C.trainRows)} flows from {C.trainMachines} machines and tested on {int(C.testRows)} flows from {C.testMachines} other machines. Switch the metric and watch the order change.</>}
    >
      <Race />
      <div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {C.models.map((r, i) => <ModelCard key={r.key} r={r} rankNo={i + 1} />)}
      </div>
      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <TuningSteps />
        <SearchGap />
        <Folds />
      </div>
    </Chapter>
  )
}
