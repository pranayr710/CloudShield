import { useEffect, useRef, useState } from 'react'
import { Chapter } from '@/components/chapter'
import { gsap, useGSAP, prefersReducedMotion } from '@/lib/gsap'
import { T, TON_COLOR, TON_NOTE, bytes, int, pct, tonLabel, type TonFlow } from '@/lib/data'
import { cn } from '@/lib/utils'

const O = T.overview
const C = T.classification
const max = Math.max(...O.classes.map((c) => c.count))

/** One bar per class, with how many source machines produced it. */
function ClassBars() {
  const root = useRef<HTMLDivElement>(null)
  const [hover, setHover] = useState('mitm')

  useGSAP(() => {
    if (prefersReducedMotion()) return
    gsap.from(root.current!.querySelectorAll('[data-bar]'), {
      scaleX: 0, duration: 1.2, stagger: 0.06, ease: 'expo.out',
      scrollTrigger: { trigger: root.current, start: 'top 80%', once: true },
    })
  }, { scope: root })

  const c = O.classes.find((x) => x.name === hover)!
  return (
    <div ref={root} className="grain grid content-start gap-6 rounded-2xl border border-hairline-strong bg-panel p-6 md:p-8">
      <div className="flex items-baseline justify-between gap-4">
        <span className="eyebrow">Flows per class</span>
        <span className="text-xs text-ink-dim">after removing duplicates · machines</span>
      </div>
      <ul className="grid gap-1.5">
        {O.classes.map((x) => (
          <li key={x.name}>
            <button type="button" onMouseEnter={() => setHover(x.name)} onFocus={() => setHover(x.name)}
              className={cn('grid w-full grid-cols-[6.5rem_minmax(0,1fr)_4.2rem_1.6rem] items-center gap-3 rounded-md px-1.5 py-1 text-left text-sm transition-colors',
                hover === x.name ? 'bg-panel-2' : '')}>
              <span className={hover === x.name ? 'font-medium text-ink' : 'text-ink-muted'}>{tonLabel(x.name)}</span>
              <span className="relative h-3 overflow-hidden rounded-sm bg-panel-2">
                <span data-bar className="absolute inset-y-0 left-0 origin-left" style={{ width: `${Math.max(1.5, (x.count / max) * 100)}%`, background: TON_COLOR[x.name] }} />
              </span>
              <span className="num text-right text-xs text-ink-muted">{int(x.count)}</span>
              <span className="num text-right text-xs text-ink-dim" title="source machines">{x.machines}</span>
            </button>
          </li>
        ))}
      </ul>
      <div className="grid gap-1 border-t border-hairline-strong pt-5">
        <div className="flex items-baseline justify-between gap-4">
          <span className="font-medium" style={{ color: TON_COLOR[hover] }}>{tonLabel(hover)}</span>
          <span className="num text-sm text-ink-muted">{pct(c.count / O.rowsClean, 2)} of flows · {c.machines} machine{c.machines > 1 ? 's' : ''}</span>
        </div>
        <p className="text-sm leading-relaxed text-ink-muted">{TON_NOTE[hover]}</p>
      </div>
    </div>
  )
}

type Row = TonFlow & { key: number; t: string }

/** Real test-set flows arrive one by one and the final model labels each of them. */
function LiveFeed() {
  const root = useRef<HTMLDivElement>(null)
  const [rows, setRows] = useState<Row[]>([])
  const [seen, setSeen] = useState({ flows: 0, attacks: 0, right: 0 })
  const cursor = useRef(0)

  useEffect(() => {
    let timer: number | undefined
    const push = () => {
      const f = C.stream[cursor.current % C.stream.length]
      const k = cursor.current++
      const now = new Date()
      const t = now.toTimeString().slice(0, 8) + '.' + String(now.getMilliseconds()).padStart(3, '0')
      setRows((r) => [{ ...f, key: k, t }, ...r].slice(0, 11))
      setSeen((s) => ({ flows: s.flows + 1, attacks: s.attacks + (f.label === 'normal' ? 0 : 1), right: s.right + (f.pred === f.label ? 1 : 0) }))
    }
    const io = new IntersectionObserver(([e]) => {
      window.clearInterval(timer)
      if (e.isIntersecting) {
        if (cursor.current === 0) for (let i = 0; i < 8; i++) push()
        timer = window.setInterval(push, prefersReducedMotion() ? 2500 : 850)
      }
    })
    io.observe(root.current!)
    return () => { io.disconnect(); window.clearInterval(timer) }
  }, [])

  return (
    <div ref={root} className="screen scanlines relative grid content-start overflow-hidden rounded-2xl">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-hairline-strong px-5 py-3.5">
        <span className="flex items-center gap-2.5">
          <span className="h-2 w-2 animate-pulse rounded-full bg-red" />
          <span className="eyebrow text-ink">Live · {C.best} on the test set</span>
        </span>
        <span className="num text-xs text-ink-muted">
          {int(seen.flows)} flows · <span className="text-red">{int(seen.attacks)} attacks</span> · <span className="text-phosphor">{int(seen.right)} right</span>
        </span>
      </div>
      <div className="overflow-x-auto">
        <table className="num w-full min-w-[640px] whitespace-nowrap text-left text-[0.75rem]">
          <thead className="text-ink-dim">
            <tr className="[&>th]:px-3 [&>th]:py-2.5 [&>th]:font-normal">
              <th>time</th><th>proto</th><th>service</th><th>state</th><th className="text-right">sent</th><th className="text-right">received</th><th>model says</th><th>truth</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => {
              const ok = r.pred === r.label
              return (
                <tr key={r.key}
                  className={cn('border-t border-hairline [&>td]:px-3 [&>td]:py-2', i === 0 && 'animate-in fade-in slide-in-from-top-2 duration-500',
                    r.label !== 'normal' ? 'text-ink' : 'text-ink-muted')}
                  style={{ opacity: 1 - i * 0.06 }}>
                  <td className="text-ink-dim">{r.t}</td>
                  <td>{r.f.proto}</td>
                  <td>{r.f.service === '-' ? '·' : r.f.service}</td>
                  <td>{r.f.conn_state}</td>
                  <td className="text-right">{bytes(r.f.src_bytes)}</td>
                  <td className="text-right">{bytes(r.f.dst_bytes)}</td>
                  <td className={r.pred === 'normal' ? 'text-phosphor' : 'text-red'}>{tonLabel(r.pred)}</td>
                  <td className={ok ? 'text-ink-dim' : 'text-amber'}>{ok ? '✓' : `✗ ${tonLabel(r.label)}`}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export function Dataset() {
  const mitm = O.classes.find((c) => c.name === 'mitm')!
  return (
    <Chapter
      id="dataset"
      index="01"
      eyebrow={`The data · ${O.name}`}
      title={<>Ten kinds<br />of traffic</>}
      deck={<>{O.name} was recorded on an IoT and IT testbed at UNSW Canberra. Each row is one network flow, described by {O.columns} columns. After dropping {int(O.duplicates)} exact duplicates, {int(O.rowsClean)} flows remain, {pct(O.attackShare)} of them attacks. The rarest, man-in-the-middle, has only {int(mitm.count)}.</>}
    >
      <div className="grid gap-4 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
        <ClassBars />
        <LiveFeed />
      </div>
    </Chapter>
  )
}
