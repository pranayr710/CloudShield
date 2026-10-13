import { useMemo, useRef, useState } from 'react'
import { Chapter } from '@/components/chapter'
import { gsap, useGSAP, prefersReducedMotion } from '@/lib/gsap'
import { CLASS_COLOR, D, int, pct } from '@/lib/data'
import { cn } from '@/lib/utils'

const C = D.classification
const L = C.confusion.labels
const N = C.confusion.norm
const RARE = ['Analysis', 'Backdoor', 'Worms']

/** Where each rare class actually ends up: its largest wrong destination. */
const absorbed = RARE.map((name) => {
  const i = L.indexOf(name)
  const row = N[i]
  let j = -1
  row.forEach((v, k) => { if (k !== i && (j < 0 || v > row[j])) j = k })
  return { name, to: L[j], share: row[j], recall: row[i] }
})

function Heatmap() {
  const root = useRef<HTMLDivElement>(null)
  const [cell, setCell] = useState<[number, number]>([L.indexOf('DoS'), L.indexOf('Exploits')])

  useGSAP(() => {
    if (prefersReducedMotion()) return
    gsap.from(root.current!.querySelectorAll('[data-cell]'), {
      opacity: 0, scale: 0.4, duration: 0.6, ease: 'back.out(2)',
      stagger: { grid: [L.length, L.length], from: 'start', amount: 0.9 },
      scrollTrigger: { trigger: root.current, start: 'top 75%', once: true },
    })
  }, { scope: root })

  const [r, c] = cell
  const v = N[r][c]
  return (
    <div ref={root} className="grid content-start gap-5 rounded-2xl border border-hairline-strong bg-panel p-5 md:p-8">
      <div className="flex items-baseline justify-between gap-4">
        <span className="eyebrow">Confusion matrix · {C.best}</span>
        <span className="text-xs text-ink-dim">rows: true class · columns: predicted</span>
      </div>
      <div className="overflow-x-auto">
        <div className="grid min-w-[520px] gap-[3px]" style={{ gridTemplateColumns: `6.5rem repeat(${L.length}, minmax(0, 1fr))` }}
          onMouseLeave={() => setCell([L.indexOf('DoS'), L.indexOf('Exploits')])}>
          <span />
          {L.map((l, j) => (
            <span key={l} className={cn('num truncate pb-1 text-center text-[0.6rem] uppercase tracking-wider', j === c ? 'text-ink' : 'text-ink-dim')} title={l}>
              {l.slice(0, 4)}
            </span>
          ))}
          {L.map((rowLabel, i) => (
            <div key={rowLabel} className="contents">
              <span className={cn('truncate pr-2 text-right text-xs leading-8', i === r ? 'text-ink' : 'text-ink-muted')}>{rowLabel}</span>
              {L.map((colLabel, j) => {
                const x = N[i][j]
                const diag = i === j
                const col = diag ? '54,226,212' : '255,59,92'
                return (
                  <button key={colLabel} type="button" data-cell
                    onMouseEnter={() => setCell([i, j])} onFocus={() => setCell([i, j])}
                    aria-label={`${pct(x, 1)} of ${rowLabel} flows predicted as ${colLabel}`}
                    className={cn('num grid aspect-square min-h-8 place-items-center rounded-[4px] text-[0.62rem] transition-[outline] outline-offset-1',
                      i === r && j === c ? 'outline outline-1 outline-ink' : '')}
                    style={{ background: `rgba(${col},${x < 0.005 ? 0.04 : 0.12 + 0.88 * Math.sqrt(x)})`, color: Math.sqrt(x) > 0.55 ? '#05080c' : 'rgba(234,242,245,0.75)' }}>
                    {x >= 0.01 ? Math.round(x * 100) : ''}
                  </button>
                )
              })}
            </div>
          ))}
        </div>
      </div>
      <p className="min-h-[3rem] border-t border-hairline-strong pt-4 text-[0.95rem] leading-relaxed" aria-live="polite">
        <span className="num" style={{ color: r === c ? '#36e2d4' : '#ff3b5c' }}>{pct(v, 1)}</span>{' '}
        of <span style={{ color: CLASS_COLOR[L[r]] }}>{L[r]}</span> flows
        {r === c ? ' were identified correctly' : <> were labelled <span style={{ color: CLASS_COLOR[L[c]] }}>{L[c]}</span></>}
        <span className="text-ink-dim"> ({int(C.confusion.counts[r][c])} of {int(C.confusion.counts[r].reduce((a, b) => a + b, 0))})</span>.
      </p>
    </div>
  )
}

type Metric = 'acc' | 'f1w' | 'f1m'
const METRICS: { key: Metric; label: string; hint: string }[] = [
  { key: 'acc', label: 'Accuracy', hint: 'Share of test flows labelled correctly. Flattered by the big classes.' },
  { key: 'f1w', label: 'Weighted F1', hint: 'F1 per class, averaged by class size. The headline metric.' },
  { key: 'f1m', label: 'Macro F1', hint: 'F1 per class, every class counting equally. Exposes the rare classes.' },
]

function Leaderboard() {
  const [metric, setMetric] = useState<Metric>('f1w')
  const rows = useMemo(() => [...C.models].sort((a, b) => b[metric] - a[metric]), [metric])
  const m = METRICS.find((x) => x.key === metric)!

  return (
    <div className="beam grid content-start gap-6 rounded-2xl bg-panel p-6 md:p-8">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <span className="eyebrow">Five classifiers, one test set</span>
        <div className="flex gap-1 rounded-full border border-hairline-strong p-1" role="tablist">
          {METRICS.map((x) => (
            <button key={x.key} type="button" role="tab" aria-selected={metric === x.key} onClick={() => setMetric(x.key)}
              className={cn('rounded-full px-3 py-1.5 text-xs font-medium transition', metric === x.key ? 'bg-cyan text-carbon' : 'text-ink-muted hover:text-ink')}>
              {x.label}
            </button>
          ))}
        </div>
      </div>
      <ol className="grid gap-3">
        {rows.map((row, i) => (
          <li key={row.model} className="grid grid-cols-[1.5rem_9.5rem_minmax(0,1fr)_3.5rem] items-center gap-3 text-sm transition-all duration-500">
            <span className="num text-ink-dim">{i + 1}</span>
            <span className={row.model === C.best ? 'text-ink' : 'text-ink-muted'}>{row.model}</span>
            <span className="relative h-3 rounded-full bg-panel-2">
              <span className={cn('absolute inset-y-0 left-0 rounded-full transition-[width] duration-700 ease-out', row.model === C.best ? 'bg-cyan shadow-[0_0_14px_#36e2d4]' : 'bg-ink/70')}
                style={{ width: `${row[metric] * 100}%` }} />
              {metric === 'acc' && <span className="absolute inset-y-[-5px] w-px bg-red" style={{ left: `${C.baseline.acc * 100}%` }} />}
            </span>
            <span className="num text-right">{row[metric].toFixed(3)}</span>
          </li>
        ))}
      </ol>
      <p className="text-sm leading-relaxed text-ink-muted">
        {m.hint}{metric === 'acc' && <> The red line is the {pct(C.baseline.acc, 1)} you get by always answering Normal.</>}
      </p>
      <div className="grid gap-2 border-t border-hairline-strong pt-5 text-sm">
        <span className="eyebrow">5-fold cross-validation, weighted F1</span>
        {C.cv.map((x) => (
          <div key={x.model} className="flex justify-between gap-4"><span className="text-ink-muted">{x.model}</span><span className="num">{x.mean.toFixed(4)} ± {x.std.toFixed(4)}</span></div>
        ))}
      </div>
    </div>
  )
}

export function Confusion() {
  return (
    <Chapter
      id="confusion"
      index="03"
      tone="red"
      eyebrow="Where the model goes wrong"
      title={<>Some attacks<br />hide in others</>}
      deck={<>Normal and Generic traffic are close to solved. The three rarest families are not: {absorbed.map((a, i) => (
        <span key={a.name}>{a.name} is recognised {pct(a.recall, 1)} of the time{i < absorbed.length - 1 ? ', ' : '. '}</span>
      ))}Every one of the five algorithms fails on the same classes, which points at the data rather than the model.</>}
    >
      <div className="grid gap-4 lg:grid-cols-[minmax(0,7fr)_minmax(0,5fr)]">
        <Heatmap />
        <Leaderboard />
      </div>
      <div className="mt-4 grid gap-px overflow-hidden rounded-2xl border border-hairline-strong bg-hairline-strong md:grid-cols-3">
        {absorbed.map((a) => (
          <div key={a.name} className="grid gap-2 bg-panel p-6">
            <span className="eyebrow">{a.name} is absorbed</span>
            <span className="num text-[2.2rem] font-medium leading-none text-red">{pct(a.share, 0)}</span>
            <span className="text-sm text-ink-muted">of {a.name} flows are labelled <span style={{ color: CLASS_COLOR[a.to] }}>{a.to}</span>. They look the same at the flow level.</span>
          </div>
        ))}
      </div>
    </Chapter>
  )
}
