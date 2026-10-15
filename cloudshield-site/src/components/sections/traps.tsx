import { useState, type ReactNode } from 'react'
import { Chapter } from '@/components/chapter'
import { Switch } from '@/components/ui/switch'
import { T, TON_COLOR, int, pct, tonLabel } from '@/lib/data'
import { cn } from '@/lib/utils'

const O = T.overview

function Card({ index, title, toggle, on, setOn, children, verdict }: {
  index: string; title: string; toggle: string; on: boolean; setOn: (v: boolean) => void
  children: ReactNode; verdict: ReactNode
}) {
  return (
    <article className={cn('grain grid content-start gap-6 rounded-2xl border bg-panel p-6 transition-colors duration-500 md:p-8', on ? 'border-red' : 'border-hairline-strong')}>
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <span className={cn('num text-sm', on ? 'text-red' : 'text-signal')}>{index}</span>
          <h3 className="display text-[1.1rem] leading-tight">{title}</h3>
        </div>
        <span className={cn('stamp transition-colors', on ? 'text-red' : 'text-signal')}>{on ? 'Leaks' : 'Closed'}</span>
      </div>
      <label className="flex cursor-pointer items-center justify-between gap-4 rounded-xl border border-hairline-strong bg-paper px-4 py-3 text-sm">
        <span>{toggle}</span>
        <Switch checked={on} onCheckedChange={setOn} className="data-[state=checked]:bg-red" />
      </label>
      {children}
      <p className={cn('text-sm leading-relaxed transition-colors', on ? 'text-red' : 'text-ink-muted')}>{verdict}</p>
    </article>
  )
}

/** One square per 1,000 flows. */
function Duplicates() {
  const [on, setOn] = useState(false)
  const total = Math.round(O.rows / 1000)
  const dup = Math.round(O.duplicates / 1000)
  return (
    <Card index="A" title="The same flow, twice" toggle="Keep duplicate rows" on={on} setOn={setOn}
      verdict={on
        ? <>{int(O.duplicates)} copies would be split at random, so many test flows would have an identical twin in training. The score would measure memory, not skill.</>
        : <>{int(O.duplicates)} exact duplicates ({pct(O.duplicates / O.rows, 1)}) are removed before the data is split.</>}>
      <div className="flex flex-wrap gap-[3px]" aria-hidden>
        {Array.from({ length: total }, (_, i) => {
          const isDup = i >= total - dup
          return <span key={i} className={cn('h-2 w-2 rounded-[1px] transition-all duration-500', isDup ? (on ? 'bg-red' : 'bg-panel-2') : 'bg-ink/70')}
            style={{ transitionDelay: `${(i % 40) * 6}ms` }} />
        })}
      </div>
      <span className="num text-xs text-ink-dim">one square = 1,000 flows</span>
    </Card>
  )
}

/** label is 0 exactly when type is normal. */
function Leak() {
  const [on, setOn] = useState(false)
  return (
    <Card index="B" title="The answer in the question" toggle="Keep the label column" on={on} setOn={setOn}
      verdict={on
        ? <><code className="num">label</code> is 0 for every normal flow and 1 for every attack, so it gives away the hardest half of the answer before the model has looked at the traffic.</>
        : <><code className="num">label</code> is dropped. The model has to tell normal from attack from the traffic itself.</>}>
      <div className="num grid gap-2 rounded-xl bg-paper p-4 text-xs">
        {[['normal', 0], ['ddos', 1], ['scanning', 1], ['mitm', 1]].map(([t, l]) => (
          <div key={t} className="grid grid-cols-[1fr_auto_auto] items-center gap-3">
            <span style={{ color: TON_COLOR[t as string] }}>type = {tonLabel(t as string)}</span>
            <span className="text-ink-dim">→</span>
            <span className={cn('rounded px-2 py-0.5 transition', on ? 'bg-red text-paper' : 'bg-panel-2 text-ink-dim line-through')}>label = {l}</span>
          </div>
        ))}
      </div>
    </Card>
  )
}

/** Each attack comes from a handful of machines. */
function Machines() {
  const [on, setOn] = useState(false)
  const few = O.classes.filter((c) => c.name !== 'normal').sort((a, b) => a.machines - b.machines)
  return (
    <Card index="C" title="Same machine, both sides" toggle="Split the flows at random" on={on} setOn={setOn}
      verdict={on
        ? <>{tonLabel(few[0].name)} and {tonLabel(few[1].name)} attacks each come from just {few[1].machines === 1 ? 'one machine' : `${few[1].machines} machines`}. Split at random, the same attacking machine sits in training and in test, and the model can recognise the machine instead of the attack.</>
        : <>Flows are split by source machine (the source IP): every test flow comes from a machine the model has never seen.</>}>
      <div className="grid gap-1.5">
        {few.map((c) => (
          <div key={c.name} className="grid grid-cols-[6rem_minmax(0,1fr)] items-center gap-3 text-xs">
            <span className="text-ink-muted">{tonLabel(c.name)}</span>
            <span className="flex gap-1">
              {Array.from({ length: c.machines }, (_, i) => (
                <span key={i} className={cn('grid h-4 w-9 place-items-center rounded-sm text-[0.55rem] transition-colors', on ? 'text-paper' : 'bg-panel-2 text-ink-dim')}
                  style={on ? { background: TON_COLOR[c.name] } : undefined}>
                  {on ? 'both' : 'one'}
                </span>
              ))}
            </span>
          </div>
        ))}
      </div>
      <span className="num text-xs text-ink-dim">source machines behind each attack type{on ? ': split at random, a machine lands on both sides' : ': grouped, each machine on one side only'}</span>
    </Card>
  )
}

export function Traps() {
  return (
    <Chapter
      id="traps"
      index="02"
      tone="red"
      eyebrow="Keeping the score honest"
      title={<>Three ways to<br />fake a result</>}
      deck={<>A near-perfect score is easy to get from this dataset and would mean nothing. Flip each switch to see a shortcut, and how the pipeline closes it.</>}
    >
      <div className="grid gap-4 lg:grid-cols-3">
        <Duplicates />
        <Leak />
        <Machines />
      </div>
    </Chapter>
  )
}
