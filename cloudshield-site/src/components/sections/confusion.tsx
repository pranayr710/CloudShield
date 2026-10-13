import { useRef, useState } from 'react'
import { Chapter } from '@/components/chapter'
import { gsap, useGSAP, prefersReducedMotion } from '@/lib/gsap'
import { T, TON_COLOR, int, pct, tonLabel } from '@/lib/data'
import { cn } from '@/lib/utils'

const C = T.classification
const L = C.confusion.labels
const N = C.confusion.norm
const worst = [...L].sort((a, b) => C.recall[a] - C.recall[b])[0]
const worstRow = N[L.indexOf(worst)]
const worstTo = L[worstRow.map((v, j) => (j === L.indexOf(worst) ? -1 : v)).reduce((bi, v, j, a) => (v > a[bi] ? j : bi), 0)]

function Heatmap() {
  const root = useRef<HTMLDivElement>(null)
  const start: [number, number] = [L.indexOf(worst), L.indexOf(worstTo)]
  const [cell, setCell] = useState<[number, number]>(start)

  useGSAP(() => {
    if (prefersReducedMotion()) return
    gsap.from(root.current!.querySelectorAll('[data-cell]'), {
      opacity: 0, scale: 0.4, duration: 0.6, ease: 'back.out(2)',
      stagger: { grid: [L.length, L.length], from: 'start', amount: 0.9 },
      scrollTrigger: { trigger: root.current, start: 'top 75%', once: true },
    })
  }, { scope: root })

  const [r, c] = cell
  const counts = C.confusion.counts
  return (
    <div ref={root} className="grain grid content-start gap-5 rounded-2xl border border-hairline-strong bg-panel p-5 md:p-8">
      <div className="flex items-baseline justify-between gap-4">
        <span className="eyebrow">Confusion matrix · {C.best} (tuned)</span>
        <span className="text-xs text-ink-dim">rows: true class · columns: predicted</span>
      </div>
      <div className="overflow-x-auto">
        <div className="grid min-w-[540px] gap-[3px]" style={{ gridTemplateColumns: `6.2rem repeat(${L.length}, minmax(0, 1fr))` }} onMouseLeave={() => setCell(start)}>
          <span />
          {L.map((l, j) => (
            <span key={l} className={cn('num truncate pb-1 text-center text-[0.6rem] uppercase tracking-wider', j === c ? 'text-ink' : 'text-ink-dim')} title={tonLabel(l)}>{tonLabel(l).slice(0, 4)}</span>
          ))}
          {L.map((rowLabel, i) => (
            <div key={rowLabel} className="contents">
              <span className={cn('truncate pr-2 text-right text-xs leading-8', i === r ? 'font-medium text-ink' : 'text-ink-muted')}>{tonLabel(rowLabel)}</span>
              {L.map((colLabel, j) => {
                const x = N[i][j]
                const col = i === j ? '15,110,90' : '229,72,28'
                const strong = Math.sqrt(x) > 0.5
                return (
                  <button key={colLabel} type="button" data-cell onMouseEnter={() => setCell([i, j])} onFocus={() => setCell([i, j])}
                    aria-label={`${pct(x, 1)} of ${tonLabel(rowLabel)} flows predicted as ${tonLabel(colLabel)}`}
                    className={cn('num grid aspect-square min-h-8 place-items-center rounded-[3px] text-[0.62rem]', i === r && j === c ? 'outline outline-2 outline-offset-1 outline-ink' : '')}
                    style={{ background: x < 0.0005 ? 'rgba(22,20,15,0.04)' : `rgba(${col},${0.12 + 0.88 * Math.sqrt(x)})`, color: strong ? '#efeae0' : 'rgba(22,20,15,0.75)' }}>
                    {x >= 0.005 ? Math.round(x * 100) : ''}
                  </button>
                )
              })}
            </div>
          ))}
        </div>
      </div>
      <p className="min-h-[3rem] border-t border-hairline-strong pt-4 text-[0.95rem] leading-relaxed" aria-live="polite">
        <span className={cn('num font-medium', r === c ? 'text-signal' : 'text-red')}>{pct(N[r][c], 1)}</span> of{' '}
        <span style={{ color: TON_COLOR[L[r]] }}>{tonLabel(L[r])}</span> flows
        {r === c ? ' were identified correctly' : <> were labelled <span style={{ color: TON_COLOR[L[c]] }}>{tonLabel(L[c])}</span></>}
        <span className="text-ink-dim"> ({int(counts[r][c])} of {int(counts[r].reduce((a, b) => a + b, 0))})</span>.
      </p>
    </div>
  )
}

function Recall() {
  const order = [...L].sort((a, b) => C.recall[a] - C.recall[b])
  return (
    <div className="grain grid content-start gap-4 rounded-2xl border border-hairline-strong bg-panel p-6 md:p-8">
      <span className="eyebrow">Recall per class, worst first</span>
      {order.map((l) => (
        <div key={l} className="grid grid-cols-[6rem_minmax(0,1fr)_3.6rem] items-center gap-3 text-sm">
          <span className={l === worst ? 'font-semibold' : 'text-ink-muted'}>{tonLabel(l)}</span>
          <span className="relative h-2.5 rounded-sm bg-panel-2">
            <span className="absolute inset-y-0 left-0 rounded-sm" style={{ width: `${C.recall[l] * 100}%`, background: TON_COLOR[l] }} />
          </span>
          <span className="num text-right">{pct(C.recall[l], 1)}</span>
        </div>
      ))}
      <p className="text-sm leading-relaxed text-ink-muted">
        {tonLabel(worst)} is the weak spot: {pct(C.recall[worst], 1)} recall, and its most common mistake is {tonLabel(worstTo)}.
      </p>
    </div>
  )
}

/** The same model configuration, retrained on some machines and tested on others. */
function StressTest() {
  const root = useRef<HTMLDivElement>(null)
  const a = useRef<HTMLSpanElement>(null)
  const b = useRef<HTMLSpanElement>(null)
  const line = useRef<SVGPathElement>(null)
  const s = C.stress

  useGSAP(() => {
    const fmt = (v: number) => `${(v * 100).toFixed(1)}%`
    if (prefersReducedMotion()) {
      a.current!.textContent = fmt(C.final.acc); b.current!.textContent = fmt(s.acc)
      return
    }
    const state = { v: C.final.acc }
    a.current!.textContent = fmt(C.final.acc)
    b.current!.textContent = fmt(C.final.acc)
    const tl = gsap.timeline({ scrollTrigger: { trigger: root.current, start: 'top 70%', once: true } })
    tl.from(line.current, { drawSVG: '0%', duration: 1.4, ease: 'power2.inOut' })
      .to(state, { v: s.acc, duration: 1.4, ease: 'power2.inOut', onUpdate: () => { b.current!.textContent = fmt(state.v) } }, 0)
  }, { scope: root })

  return (
    <div ref={root} className="screen grain relative grid gap-8 overflow-hidden rounded-2xl p-6 md:grid-cols-[minmax(0,5fr)_minmax(0,7fr)] md:p-10">
      <div className="grid content-start gap-4">
        <span className="eyebrow text-red">Stress test</span>
        <h3 className="display text-[clamp(1.6rem,3.4vw,2.6rem)] leading-[1.05]">Same model.<br />New machines.</h3>
        <p className="text-sm leading-relaxed text-ink-muted">
          The {C.best} configuration is retrained on flows from some source machines and tested only on machines it has never seen.
          Each attack in {T.overview.name} comes from two to six machines, so this is a much harder exam.
        </p>
        <p className="num text-xs text-ink-dim">classes present in that fold: {s.classes.map(tonLabel).join(', ')}</p>
      </div>
      <div className="grid content-center gap-6">
        <svg viewBox="0 0 400 120" className="w-full" aria-hidden>
          <path ref={line} d="M10 20 C 150 20, 220 22, 260 60 S 340 100, 390 100" fill="none" stroke="#e5481c" strokeWidth="3" strokeLinecap="round" />
          <circle cx="10" cy="20" r="6" fill="#5fd39a" />
          <circle cx="390" cy="100" r="6" fill="#e5481c" />
        </svg>
        <div className="grid grid-cols-2 gap-6">
          <div className="grid gap-1">
            <span ref={a} className="num text-[clamp(2.2rem,5vw,3.8rem)] font-medium leading-none text-phosphor" />
            <span className="eyebrow">accuracy, machines seen in training</span>
          </div>
          <div className="grid justify-items-end gap-1 text-right">
            <span ref={b} className="num text-[clamp(2.2rem,5vw,3.8rem)] font-medium leading-none text-red" />
            <span className="eyebrow">accuracy, unseen machines</span>
          </div>
        </div>
        <p className="num text-xs text-ink-dim">unseen machines: weighted F1 {s.f1w.toFixed(4)} · macro F1 {s.f1m.toFixed(4)}</p>
      </div>
    </div>
  )
}

export function Confusion() {
  return (
    <Chapter
      id="errors"
      index="05"
      tone="red"
      eyebrow="Where the final model goes wrong"
      title={<>Almost every<br />attack, named</>}
      deck={<>{L.filter((l) => C.recall[l] > 0.96).length} of the ten classes are recognised more than 96% of the time. The exception is the rarest one, and the real test is a network the model has never seen.</>}
    >
      <div className="grid gap-4 lg:grid-cols-[minmax(0,7fr)_minmax(0,5fr)]">
        <Heatmap />
        <Recall />
      </div>
      <div className="mt-4">
        <StressTest />
      </div>
    </Chapter>
  )
}
