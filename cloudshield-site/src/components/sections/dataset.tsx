import { useEffect, useRef, useState } from 'react'
import { Chapter } from '@/components/chapter'
import { gsap, useGSAP, prefersReducedMotion } from '@/lib/gsap'
import { CLASS_COLOR, CLASS_NOTE, D, bytes, int, pct, type Flow } from '@/lib/data'
import { cn } from '@/lib/utils'

const O = D.overview
const STREAM = D.classification.stream
const max = Math.max(...O.classCounts.map((c) => c.count))

/** Horizontal bars, one per class, on a square-root scale so Worms is still visible. */
function ClassBars() {
  const root = useRef<HTMLDivElement>(null)
  const [hover, setHover] = useState<string>('Normal')

  useGSAP(() => {
    if (prefersReducedMotion()) return
    gsap.from(root.current!.querySelectorAll('[data-bar]'), {
      scaleX: 0, duration: 1.2, stagger: 0.06, ease: 'expo.out',
      scrollTrigger: { trigger: root.current, start: 'top 80%', once: true },
    })
  }, { scope: root })

  const c = O.classCounts.find((x) => x.name === hover)!
  return (
    <div ref={root} className="grid content-start gap-6 rounded-2xl border border-hairline-strong bg-panel p-6 md:p-8">
      <div className="flex items-baseline justify-between gap-4">
        <span className="eyebrow">Flows per class</span>
        <span className="text-xs text-ink-dim">square-root scale</span>
      </div>
      <ul className="grid gap-2" onMouseLeave={() => setHover('Normal')}>
        {O.classCounts.map((x) => (
          <li key={x.name}>
            <button type="button" onMouseEnter={() => setHover(x.name)} onFocus={() => setHover(x.name)}
              className={cn('grid w-full grid-cols-[7.5rem_minmax(0,1fr)_4.5rem] items-center gap-3 rounded-md px-1 py-1 text-left text-sm transition-colors',
                hover === x.name ? 'bg-white/[0.04]' : '')}>
              <span className={hover === x.name ? 'text-ink' : 'text-ink-muted'}>{x.name}</span>
              <span className="relative h-3 overflow-hidden rounded-full bg-panel-2">
                <span data-bar className="absolute inset-y-0 left-0 origin-left rounded-full"
                  style={{ width: `${Math.max(1.2, Math.sqrt(x.count / max) * 100)}%`, background: CLASS_COLOR[x.name],
                    boxShadow: hover === x.name ? `0 0 16px ${CLASS_COLOR[x.name]}` : 'none' }} />
              </span>
              <span className="num text-right text-xs text-ink-muted">{int(x.count)}</span>
            </button>
          </li>
        ))}
      </ul>
      <div className="grid gap-1 border-t border-hairline-strong pt-5">
        <div className="flex items-baseline justify-between gap-4">
          <span className="font-medium" style={{ color: CLASS_COLOR[hover] }}>{hover}</span>
          <span className="num text-sm text-ink-muted">{pct(c.count / O.rows, 2)} of flows</span>
        </div>
        <p className="text-sm leading-relaxed text-ink-muted">{CLASS_NOTE[hover]}</p>
      </div>
    </div>
  )
}

type Row = Flow & { key: number; t: string }

/** A live-looking packet log: real test-set flows scroll in while the panel is on screen. */
function PacketTerminal() {
  const root = useRef<HTMLDivElement>(null)
  const [rows, setRows] = useState<Row[]>([])
  const [seen, setSeen] = useState({ flows: 0, attacks: 0 })
  const cursor = useRef(0)

  useEffect(() => {
    let timer: number | undefined
    const push = () => {
      const f = STREAM[cursor.current % STREAM.length]
      const k = cursor.current++
      const now = new Date()
      const t = now.toTimeString().slice(0, 8) + '.' + String(now.getMilliseconds()).padStart(3, '0')
      setRows((r) => [{ ...f, key: k, t }, ...r].slice(0, 11))
      setSeen((s) => ({ flows: s.flows + 1, attacks: s.attacks + (f.label === 'Normal' ? 0 : 1) }))
    }
    const io = new IntersectionObserver(([e]) => {
      window.clearInterval(timer)
      if (e.isIntersecting) {
        if (cursor.current === 0) for (let i = 0; i < 8; i++) push()
        timer = window.setInterval(push, prefersReducedMotion() ? 2500 : 900)
      }
    })
    io.observe(root.current!)
    return () => { io.disconnect(); window.clearInterval(timer) }
  }, [])

  return (
    <div ref={root} className="scanlines relative grid content-start overflow-hidden rounded-2xl border border-hairline-strong bg-[#03060a]">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-hairline-strong px-5 py-3.5">
        <span className="flex items-center gap-2.5">
          <span className="h-2 w-2 animate-pulse rounded-full bg-cyan shadow-[0_0_10px_#36e2d4]" />
          <span className="eyebrow text-ink">Live capture · test set replay</span>
        </span>
        <span className="num text-xs text-ink-muted">
          {int(seen.flows)} flows · <span className="text-red">{int(seen.attacks)} attacks</span>
        </span>
      </div>
      <div className="overflow-x-auto">
        <table className="num w-full min-w-[620px] whitespace-nowrap text-left text-[0.75rem]">
          <thead className="text-ink-dim">
            <tr className="[&>th]:px-3 [&>th]:py-2.5 [&>th]:font-normal">
              <th>time</th><th>proto</th><th>service</th><th>state</th><th className="text-right">sent</th><th className="text-right">received</th><th>ttl</th><th>label</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => {
              const attack = r.label !== 'Normal'
              return (
                <tr key={r.key}
                  className={cn('border-t border-hairline transition-colors duration-700 [&>td]:px-3 [&>td]:py-2',
                    i === 0 && 'animate-in fade-in slide-in-from-top-2 duration-500',
                    attack ? 'bg-red/[0.06] text-ink' : 'text-ink-muted')}
                  style={{ opacity: 1 - i * 0.06 }}>
                  <td className="text-ink-dim">{r.t}</td>
                  <td>{r.f.proto}</td>
                  <td>{r.f.service === '-' ? '·' : r.f.service}</td>
                  <td>{r.f.state}</td>
                  <td className="text-right">{bytes(r.f.sbytes)}</td>
                  <td className="text-right">{bytes(r.f.dbytes)}</td>
                  <td>{r.f.sttl}</td>
                  <td>
                    <span className="inline-flex items-center gap-1.5" style={{ color: CLASS_COLOR[r.label] }}>
                      <span className="h-1.5 w-1.5 rounded-full" style={{ background: CLASS_COLOR[r.label] }} />
                      {r.label}
                    </span>
                  </td>
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
  return (
    <Chapter
      id="dataset"
      index="01"
      eyebrow={`The data · ${O.dataset.name}`}
      title={<>Ten kinds<br />of traffic</>}
      deck={<>Recorded on a cyber-range testbed in Canberra that mixed real background traffic with nine families of synthetic attacks. Each row is one flow, described by {O.features} statistics. Normal traffic outnumbers Worms {O.imbalance} to 1.</>}
    >
      <div className="grid gap-4 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
        <ClassBars />
        <PacketTerminal />
      </div>
    </Chapter>
  )
}
