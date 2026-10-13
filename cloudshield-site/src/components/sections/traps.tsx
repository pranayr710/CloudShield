import { useState } from 'react'
import { Chapter } from '@/components/chapter'
import { Switch } from '@/components/ui/switch'
import { D, int, pct } from '@/lib/data'
import { cn } from '@/lib/utils'

const O = D.overview
const R = D.regression
const C = D.classification
const worms = O.classCounts.find((c) => c.name === 'Worms')!.count
const normal = O.classCounts.find((c) => c.name === 'Normal')!.count

function Card({ index, title, toggle, on, setOn, children, verdict }: {
  index: string; title: string; toggle: string; on: boolean; setOn: (v: boolean) => void
  children: React.ReactNode; verdict: React.ReactNode
}) {
  return (
    <article className={cn('grid content-start gap-6 rounded-2xl border bg-panel p-6 transition-colors duration-500 md:p-8', on ? 'border-red/50' : 'border-hairline-strong')}>
      <div className="flex items-center gap-3">
        <span className={cn('num text-sm', on ? 'text-red' : 'text-cyan')}>{index}</span>
        <h3 className="display text-[1.15rem] leading-tight">{title}</h3>
      </div>
      <label className="flex cursor-pointer items-center justify-between gap-4 rounded-xl border border-hairline-strong px-4 py-3 text-sm">
        <span>{toggle}</span>
        <Switch checked={on} onCheckedChange={setOn} className="data-[state=checked]:bg-red" />
      </label>
      {children}
      <p className={cn('text-sm leading-relaxed transition-colors', on ? 'text-red' : 'text-ink-muted')}>{verdict}</p>
    </article>
  )
}

/** One dot per 1,000 flows; with duplicates kept, the copies show up in the test set too. */
function Duplicates() {
  const [on, setOn] = useState(false)
  const total = Math.round(O.rows / 1000)
  const dup = Math.round(O.duplicates / 1000)
  return (
    <Card index="A" title="Copies in the exam" toggle="Pool both files, split at random" on={on} setOn={setOn}
      verdict={on
        ? <>{pct(O.contamination, 1)} of test rows would be exact copies of training rows. The model would be graded on questions it had already seen.</>
        : <>{pct(O.duplicateShare, 1)} of all rows are exact duplicates. The models keep the official train and test files, which were captured separately, so only {pct(O.officialOverlap, 1)} of test flows also occur in training.</>}>
      <div className="flex flex-wrap gap-[3px]" aria-hidden>
        {Array.from({ length: total }, (_, i) => {
          const isDup = i >= total - dup
          return <span key={i} className={cn('h-2 w-2 rounded-[2px] transition-all duration-500',
            isDup ? (on ? 'bg-red' : 'bg-white/[0.06]') : 'bg-cyan/70')} style={{ transitionDelay: `${(i % 40) * 6}ms` }} />
        })}
      </div>
      <span className="num text-xs text-ink-dim">one square = 1,000 flows</span>
    </Card>
  )
}

/** dmean is bytes per packet, so dmean × dpkts is the target itself. */
function Leakage() {
  const [on, setOn] = useState(false)
  return (
    <Card index="B" title="The answer in the question" toggle="Allow dmean and dload" on={on} setOn={setOn}
      verdict={on
        ? <>dmean × dpkts rebuilds the target with correlation {R.leak.rebuild.toFixed(3)}. Any model would score perfectly by doing arithmetic, and learn nothing.</>
        : <>Both columns are formulas of the target, so they are removed. The forest has to learn the volume from honest signals.</>}>
      <div className="num grid gap-3 rounded-xl bg-[#03060a] p-5 text-sm">
        <div className="flex flex-wrap items-center gap-2">
          <span className={cn('rounded-md px-2 py-1 transition', on ? 'bg-red/20 text-red' : 'bg-white/[0.04] text-ink-dim line-through')}>dmean</span>
          <span className="text-ink-dim">×</span>
          <span className="rounded-md bg-white/[0.06] px-2 py-1 text-ink">dpkts</span>
          <span className="text-ink-dim">=</span>
          <span className={cn('rounded-md px-2 py-1 transition', on ? 'bg-red text-carbon' : 'bg-white/[0.04] text-ink-muted')}>dbytes</span>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <span className={cn('rounded-md px-2 py-1 transition', on ? 'bg-red/20 text-red' : 'bg-white/[0.04] text-ink-dim line-through')}>dload</span>
          <span className="text-ink-dim">=</span>
          <span className="text-ink-muted">dbytes × 8 ÷ dur</span>
        </div>
      </div>
      <span className="num text-xs text-ink-dim">dmean: mean bytes per returned packet · dload: return bits per second</span>
    </Card>
  )
}

/** One dot per Worms flow's worth of Normal flows. */
function Imbalance() {
  const [on, setOn] = useState(false)
  const ratio = Math.round(normal / worms)
  return (
    <Card index="C" title="The lazy model" toggle="Judge by accuracy alone" on={on} setOn={setOn}
      verdict={on
        ? <>A model that always says Normal gets {pct(C.baseline.acc, 1)} of the test set right and catches no attacks. Accuracy rewards ignoring rare classes.</>
        : <>Weighted and macro F1 are reported next to accuracy, and class weights make mistakes on rare classes cost more.</>}>
      <div className="flex flex-wrap gap-[3px]" aria-hidden>
        {Array.from({ length: ratio + 1 }, (_, i) => (
          <span key={i} className={cn('h-1.5 w-1.5 rounded-full transition-colors duration-500',
            i === ratio ? 'bg-red shadow-[0_0_8px_#ff3b5c]' : on ? 'bg-cyan' : 'bg-cyan/50')} />
        ))}
      </div>
      <span className="num text-xs text-ink-dim">{int(normal)} Normal flows for every {int(worms)} Worms: {ratio} to 1</span>
    </Card>
  )
}

export function Traps() {
  return (
    <Chapter
      id="traps"
      index="06"
      tone="red"
      eyebrow="Doing it honestly"
      title={<>Three ways to<br />fake a result</>}
      deck={<>A near-perfect score is easy to get from this dataset and means nothing. Flip each switch to see the shortcut we closed.</>}
    >
      <div className="grid gap-4 lg:grid-cols-3">
        <Duplicates />
        <Leakage />
        <Imbalance />
      </div>
    </Chapter>
  )
}
