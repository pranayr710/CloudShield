import { useEffect, useRef, useState } from 'react'
import { Chapter } from '@/components/chapter'
import { Video } from '@/components/video'
import { gsap, useGSAP, prefersReducedMotion } from '@/lib/gsap'
import { T, TON_COLOR, TON_NOTE, bytes, int, pct, tonLabel, type TonFlow } from '@/lib/data'
import { cn } from '@/lib/utils'

const C = T.classification

/** Rotating radar sweep. Spins fast while a flow is being classified. */
function Radar({ scanning, color }: { scanning: boolean; color: string }) {
  const sweep = useRef<SVGGElement>(null)
  const spin = useRef<gsap.core.Tween | null>(null)

  useGSAP(() => {
    spin.current = gsap.to(sweep.current, { rotation: 360, svgOrigin: '100 100', duration: 4, ease: 'none', repeat: -1 })
    if (prefersReducedMotion()) spin.current.pause()
  })
  useEffect(() => {
    if (!spin.current || prefersReducedMotion()) return
    gsap.to(spin.current, { timeScale: scanning ? 5 : 1, duration: 0.4 })
  }, [scanning])

  return (
    <svg viewBox="0 0 200 200" className="w-full max-w-[280px]" aria-hidden>
      <defs>
        <linearGradient id="radar-sweep" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor={color} stopOpacity="0" />
          <stop offset="100%" stopColor={color} stopOpacity="0.55" />
        </linearGradient>
      </defs>
      {[24, 48, 72, 96].map((r) => <circle key={r} cx="100" cy="100" r={r} fill="none" stroke="rgba(239,234,224,0.16)" />)}
      <path d="M100 4V196M4 100H196" stroke="rgba(239,234,224,0.1)" />
      <g ref={sweep}>
        <path d="M100 100 L100 4 A96 96 0 0 1 183 52 Z" fill="url(#radar-sweep)" />
        <path d="M100 100 L183 52" stroke={color} strokeWidth="1.2" />
      </g>
      <circle cx="100" cy="100" r="4" fill={color} style={{ transition: 'fill 0.4s' }} />
    </svg>
  )
}

function Fact({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex items-baseline justify-between gap-4 border-b border-hairline py-2 text-sm">
      <dt className="text-ink-muted">{k}</dt>
      <dd className="num text-ink">{v}</dd>
    </div>
  )
}

/** A light-on-dark colour for a class name shown on the radar screen. */
const onScreen = (c: string) => (c === 'ransomware' ? '#efeae0' : c === 'backdoor' ? '#c9a27e' : TON_COLOR[c])

export function ClassifierLab() {
  const [flow, setFlow] = useState<TonFlow>(() => C.lab.find((f) => f.label === 'scanning' && f.pred === f.label) ?? C.lab[0])
  const [scanning, setScanning] = useState(false)
  const [scanned, setScanned] = useState(true)
  const predRef = useRef<HTMLDivElement>(null)
  const cycle = useRef<Record<string, number>>({})
  const timer = useRef<number | undefined>(undefined)

  const pick = (cls: string | null) => {
    const pool = cls ? C.lab.filter((f) => f.label === cls) : C.lab
    let next: TonFlow
    if (cls) {
      const i = cycle.current[cls] ?? 0
      cycle.current[cls] = (i + 1) % pool.length
      next = pool[i]
    } else next = pool[Math.floor(Math.random() * pool.length)]
    setFlow(next)
    setScanning(true)
    setScanned(false)
    window.clearTimeout(timer.current)
    timer.current = window.setTimeout(() => { setScanning(false); setScanned(true) }, prefersReducedMotion() ? 0 : 1100)
  }
  useEffect(() => () => window.clearTimeout(timer.current), [])

  useEffect(() => {
    if (!scanned || prefersReducedMotion() || !predRef.current) return
    gsap.fromTo(predRef.current, { opacity: 0.3 }, {
      opacity: 1, duration: 0.6, ease: 'none',
      scrambleText: { text: tonLabel(flow.pred), chars: 'upperCase', speed: 0.9 },
    })
  }, [scanned, flow])

  const correct = flow.pred === flow.label
  const color = scanned ? onScreen(flow.pred) : '#5fd39a'
  const f = flow.f

  return (
    <Chapter
      id="classifier"
      index="03"
      tone="red"
      eyebrow={`Classification · ${C.best}`}
      title={<>Name the<br />attack</>}
      deck={<>Pick a flow the model has never seen. The final model, a tuned {C.best}, reads its {C.features} encoded features and gives a probability for each of the ten classes. Over all {int(C.testRows)} test flows it is right {pct(C.final.acc, 2)} of the time.</>}
    >
      <div className="mb-6 flex flex-wrap gap-2" role="group" aria-label="Choose a flow by its true class">
        {C.order.map((c) => (
          <button key={c} type="button" onClick={() => pick(c)}
            className={cn('flex items-center gap-2 rounded-full border px-3.5 py-2 text-sm font-medium transition',
              flow.label === c ? 'border-ink bg-ink text-paper' : 'border-hairline-strong text-ink-muted hover:border-ink hover:text-ink')}>
            <span className="h-2 w-2 rounded-full" style={{ background: TON_COLOR[c], boxShadow: flow.label === c ? '0 0 0 1.5px #efeae0' : undefined }} />{tonLabel(c)}
          </button>
        ))}
        <button type="button" onClick={() => pick(null)} className="rounded-full bg-red px-4 py-2 text-sm font-semibold text-paper transition hover:brightness-110">
          Random flow
        </button>
      </div>

      <div className="grid gap-px overflow-hidden rounded-2xl border border-hairline-strong bg-hairline-strong lg:grid-cols-[minmax(0,3fr)_minmax(0,5fr)_minmax(0,4fr)]">
        <div className="bg-panel p-6 md:p-8">
          <div className="mb-3 flex items-baseline justify-between">
            <span className="eyebrow">Flow #{flow.id}</span>
            <span className="text-xs text-ink-dim">test set</span>
          </div>
          <dl>
            <Fact k="Protocol" v={f.proto} />
            <Fact k="Service" v={f.service === '-' ? 'none' : f.service} />
            <Fact k="Connection state" v={f.conn_state} />
            <Fact k="Destination port" v={String(f.dst_port)} />
            <Fact k="Duration" v={`${f.duration.toFixed(4)} s`} />
            <Fact k="Packets sent / back" v={`${int(f.src_pkts)} / ${int(f.dst_pkts)}`} />
            <Fact k="Bytes sent" v={bytes(f.src_bytes)} />
            <Fact k="Bytes back" v={bytes(f.dst_bytes)} />
          </dl>
        </div>

        <div className="screen relative grid place-items-center gap-6 overflow-hidden p-6 md:p-10">
          <div className="pointer-events-none absolute inset-0 opacity-25 mix-blend-screen">
            <Video video="attackMap" cover />
          </div>
          <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_at_center,transparent_30%,#15130f_85%)]" />
          <div className="relative grid w-full justify-items-center gap-6">
            <Radar scanning={scanning} color={color} />
            <div className="grid justify-items-center gap-2 text-center" aria-live="polite">
              <span className="eyebrow">{scanning ? 'Reading the flow' : 'Model says'}</span>
              {/* keyed so React remounts the node instead of fighting the scramble over its text */}
              <div key={`${flow.id}-${scanned}`} ref={predRef} className="display text-[clamp(2rem,5vw,3.6rem)] leading-none" style={{ color }}>
                {scanning ? '· · ·' : tonLabel(flow.pred)}
              </div>
              {scanned && (
                <span className={cn('stamp', correct ? 'text-phosphor' : 'text-red')}>
                  {correct ? 'Correct' : `Wrong · really ${tonLabel(flow.label)}`}
                </span>
              )}
            </div>
          </div>
        </div>

        <div className="grid content-start gap-6 bg-panel p-6 md:p-8">
          <div className="grid gap-3">
            <div className="flex items-baseline justify-between gap-4">
              <span className="eyebrow">Class probabilities</span>
              <span className="text-xs text-ink-dim">top four</span>
            </div>
            {flow.proba.map((p) => (
              <div key={p.name} className="grid grid-cols-[6.5rem_minmax(0,1fr)_3.2rem] items-center gap-3 text-sm">
                <span className="text-ink-muted">{tonLabel(p.name)}</span>
                <span className="relative h-2.5 overflow-hidden rounded-sm bg-panel-2">
                  <span className="absolute inset-y-0 left-0 transition-[width] duration-700 ease-out" style={{ width: scanned ? `${p.p * 100}%` : '0%', background: TON_COLOR[p.name] }} />
                </span>
                <span className="num text-right">{pct(p.p, p.p < 0.1 ? 1 : 0)}</span>
              </div>
            ))}
          </div>
          <div className="grid gap-2 border-t border-hairline-strong pt-5">
            <span className="eyebrow">True class</span>
            <p className="font-medium" style={{ color: TON_COLOR[flow.label] }}>{tonLabel(flow.label)}</p>
            <p className="text-sm leading-relaxed text-ink-muted">{TON_NOTE[flow.label]}</p>
            <p className="num text-xs text-ink-dim">this class is recalled {pct(C.recall[flow.label], 1)} of the time on the test set</p>
          </div>
          <p className="text-xs leading-relaxed text-ink-dim">
            Each class has two flows the model got right and, where one exists, one it got wrong. Click a class again for the next.
          </p>
        </div>
      </div>
    </Chapter>
  )
}
