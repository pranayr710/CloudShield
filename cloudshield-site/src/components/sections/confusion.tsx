import { useRef, useState } from 'react'
import { Chapter } from '@/components/chapter'
import { gsap, useGSAP, prefersReducedMotion } from '@/lib/gsap'
import { T, TON_COLOR, int, pct, tonLabel } from '@/lib/data'
import { cn } from '@/lib/utils'

const C = T.classification
const R = C.confusion.rows
const K = C.confusion.cols
const N = C.confusion.norm
const worst = [...R].sort((a, b) => C.recall[a] - C.recall[b])[0]
const wi = R.indexOf(worst)
/** the class the worst one is most often mistaken for */
const worstTo = K.filter((k) => k !== worst).reduce((b, k) => (N[wi][K.indexOf(k)] > N[wi][K.indexOf(b)] ? k : b))

function Heatmap() {
  const root = useRef<HTMLDivElement>(null)
  const start: [number, number] = [wi, K.indexOf(worstTo)]
  const [cell, setCell] = useState<[number, number]>(start)

  useGSAP(() => {
    if (prefersReducedMotion()) return
    gsap.from(root.current!.querySelectorAll('[data-cell]'), {
      opacity: 0, scale: 0.4, duration: 0.6, ease: 'back.out(2)',
      stagger: { grid: [R.length, K.length], from: 'start', amount: 0.8 },
      scrollTrigger: { trigger: root.current, start: 'top 75%', once: true },
    })
  }, { scope: root })

  const [r, c] = cell
  const same = R[r] === K[c]
  return (
    <div ref={root} className="grain grid content-start gap-5 rounded-2xl border border-hairline-strong bg-panel p-5 md:p-8">
      <div className="flex items-baseline justify-between gap-4">
        <span className="eyebrow">Confusion matrix · {C.bestDisplay}</span>
        <span className="text-xs text-ink-dim">rows: true class · columns: predicted</span>
      </div>
      <div className="overflow-x-auto">
        <div className="grid min-w-[520px] gap-[3px]" style={{ gridTemplateColumns: `6.2rem repeat(${K.length}, minmax(0, 1fr))` }} onMouseLeave={() => setCell(start)}>
          <span />
          {K.map((l, j) => (
            <span key={l} className={cn('num truncate pb-1 text-center text-[0.6rem] uppercase tracking-wider', j === c ? 'text-ink' : 'text-ink-dim')} title={tonLabel(l)}>{tonLabel(l).slice(0, 4)}</span>
          ))}
          {R.map((rowLabel, i) => (
            <div key={rowLabel} className="contents">
              <span className={cn('truncate pr-2 text-right text-xs leading-9', i === r ? 'font-medium text-ink' : 'text-ink-muted')}>{tonLabel(rowLabel)}</span>
              {K.map((colLabel, j) => {
                const x = N[i][j]
                const diag = rowLabel === colLabel
                return (
                  <button key={colLabel} type="button" data-cell onMouseEnter={() => setCell([i, j])} onFocus={() => setCell([i, j])}
                    aria-label={`${pct(x, 1)} of ${tonLabel(rowLabel)} flows predicted as ${tonLabel(colLabel)}`}
                    className={cn('num grid min-h-9 place-items-center rounded-[3px] text-[0.62rem]', i === r && j === c ? 'outline outline-2 outline-offset-1 outline-ink' : '')}
                    style={{ background: x < 0.0005 ? 'rgba(22,20,15,0.04)' : `rgba(${diag ? '15,110,90' : '229,72,28'},${0.12 + 0.88 * Math.sqrt(x)})`, color: Math.sqrt(x) > 0.5 ? '#efeae0' : 'rgba(22,20,15,0.75)' }}>
                    {x >= 0.005 ? Math.round(x * 100) : ''}
                  </button>
                )
              })}
            </div>
          ))}
        </div>
      </div>
      <p className="min-h-[3rem] border-t border-hairline-strong pt-4 text-[0.95rem] leading-relaxed" aria-live="polite">
        <span className={cn('num font-medium', same ? 'text-signal' : 'text-red')}>{pct(N[r][c], 1)}</span> of{' '}
        <span style={{ color: TON_COLOR[R[r]] }}>{tonLabel(R[r])}</span> flows
        {same ? ' were identified correctly' : <> were labelled <span style={{ color: TON_COLOR[K[c]] }}>{tonLabel(K[c])}</span></>}
        <span className="text-ink-dim"> ({int(C.confusion.counts[r][c])} of {int(C.confusion.counts[r].reduce((a, b) => a + b, 0))})</span>.
      </p>
    </div>
  )
}

function PerClass() {
  const rows = [...C.perClass].sort((a, b) => a.recall - b.recall)
  return (
    <div className="grain grid content-start gap-4 rounded-2xl border border-hairline-strong bg-panel p-6 md:p-8">
      <div className="flex items-baseline justify-between gap-3">
        <span className="eyebrow">Per class, worst recall first</span>
        <span className="num text-[0.65rem] text-ink-dim">precision · recall · F1</span>
      </div>
      {rows.map((p) => (
        <div key={p.name} className="grid gap-1">
          <div className="flex items-baseline justify-between text-sm">
            <span className={p.name === worst ? 'font-semibold' : 'text-ink-muted'} style={{ color: p.name === worst ? TON_COLOR[p.name] : undefined }}>{tonLabel(p.name)}</span>
            <span className="num text-xs text-ink-muted">{p.precision.toFixed(2)} · <b className="text-ink">{p.recall.toFixed(2)}</b> · {p.f1.toFixed(2)}</span>
          </div>
          <span className="relative h-2 rounded-sm bg-panel-2"><span className="absolute inset-y-0 left-0 rounded-sm" style={{ width: `${p.recall * 100}%`, background: TON_COLOR[p.name] }} /></span>
        </div>
      ))}
      <p className="text-sm leading-relaxed text-ink-muted">
        {tonLabel(worst)} has the lowest recall: {pct(C.recall[worst], 1)} of its flows from the unseen machines are recognised; the misses are most often labelled {tonLabel(worstTo)}.
      </p>
    </div>
  )
}

/** Why the test set is grouped by machine, and what that leaves out. */
function UnseenMachines() {
  const missing = T.overview.classes.filter((c) => C.missingFromTest.includes(c.name))
  return (
    <div className="screen grain grid gap-8 overflow-hidden rounded-2xl p-6 md:grid-cols-[minmax(0,5fr)_minmax(0,7fr)] md:p-10">
      <div className="grid content-start gap-4">
        <span className="eyebrow text-red">How it was tested</span>
        <h3 className="display text-[clamp(1.6rem,3.4vw,2.6rem)] leading-[1.05]">New machines<br />only.</h3>
        <p className="text-sm leading-relaxed text-ink-muted">
          The flows were split by source machine: {C.testMachines} machines were held out for testing, chosen so every traffic type appears on both sides, and every flow from a machine goes to training or to test, never both.
          A random split would let the model recognise machines it has already seen; this one measures how it would do on a new network.
        </p>
      </div>
      <div className="grid content-center gap-5">
        <div className="grid grid-cols-2 gap-4">
          <div className="grid gap-1"><span className="num text-[clamp(2rem,4.5vw,3.4rem)] leading-none text-phosphor">{C.trainMachines}</span><span className="eyebrow">machines in training · {int(C.trainRows)} flows</span></div>
          <div className="grid gap-1"><span className="num text-[clamp(2rem,4.5vw,3.4rem)] leading-none text-red">{C.testMachines}</span><span className="eyebrow">unseen machines in test · {int(C.testRows)} flows</span></div>
        </div>
        {missing.length > 0 && (
          <p className="text-sm leading-relaxed text-ink-muted">
            Because each attack comes from only a few machines, {missing.map((c) => tonLabel(c.name)).join(', ')} {missing.length > 1 ? 'have' : 'has'} no flows in the test fold:
            all of {missing.length > 1 ? 'their' : 'its'} machines landed in training. The model still learns them, but this test cannot score them.
          </p>
        )}
        <p className="num text-xs text-ink-dim">final model on the unseen machines: accuracy {C.final.acc.toFixed(4)} · weighted precision {C.final.precision.toFixed(4)} · weighted recall {C.final.recall.toFixed(4)} · ROC-AUC (OvR) {C.final.auc.toFixed(4)}</p>
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
      title={<>Hard questions,<br />honest answers</>}
      deck={<>On machines it has never seen, the {C.bestDisplay} model reaches weighted F1 {C.final.f1w.toFixed(3)}. {C.order.filter((c) => C.recall[c] >= 0.9).length} of the {C.order.length} traffic types are recalled at 90% or better; the weakest is {tonLabel(worst)} at {pct(C.recall[worst], 1)}.</>}
    >
      <div className="grid gap-4 lg:grid-cols-[minmax(0,7fr)_minmax(0,5fr)]">
        <Heatmap />
        <PerClass />
      </div>
      <div className="mt-4">
        <UnseenMachines />
      </div>
    </Chapter>
  )
}
