import { useEffect, useRef, useState } from 'react'
import { Chapter } from '@/components/chapter'
import { Video } from '@/components/video'
import { gsap, useGSAP, prefersReducedMotion } from '@/lib/gsap'
import { CLASS_COLOR, CLASS_NOTE, D, bytes, pct, type Flow } from '@/lib/data'
import { cn } from '@/lib/utils'

const C = D.classification
const best = C.models.find((m) => m.model === C.best) ?? C.models[0]

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
    <svg viewBox="0 0 200 200" className="w-full max-w-[300px]" aria-hidden>
      <defs>
        <radialGradient id="radar-glow">
          <stop offset="0%" stopColor={color} stopOpacity="0.2" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </radialGradient>
        <linearGradient id="radar-sweep" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor={color} stopOpacity="0" />
          <stop offset="100%" stopColor={color} stopOpacity="0.55" />
        </linearGradient>
      </defs>
      <circle cx="100" cy="100" r="96" fill="url(#radar-glow)" />
      {[24, 48, 72, 96].map((r) => <circle key={r} cx="100" cy="100" r={r} fill="none" stroke="rgba(160,220,255,0.14)" />)}
      <path d="M100 4V196M4 100H196" stroke="rgba(160,220,255,0.1)" />
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

export function ClassifierLab() {
  const [flow, setFlow] = useState<Flow>(() => C.lab.find((f) => f.label === 'Exploits' && f.pred === f.label) ?? C.lab[0])
  const [scanning, setScanning] = useState(false)
  const [scanned, setScanned] = useState(true)
  const predRef = useRef<HTMLDivElement>(null)
  const cycle = useRef<Record<string, number>>({})
  const timer = useRef<number | undefined>(undefined)

  const pick = (cls: string | null) => {
    const pool = cls ? C.lab.filter((f) => f.label === cls) : C.lab
    let next: Flow
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
      scrambleText: { text: flow.pred, chars: 'upperCase', speed: 0.9 },
    })
  }, [scanned, flow])

  const correct = flow.pred === flow.label
  const color = scanned ? CLASS_COLOR[flow.pred] : '#36e2d4'
  const f = flow.f

  return (
    <Chapter
      id="classifier"
      index="02"
      tone="red"
      eyebrow={`Classification · ${C.best}`}
      title={<>Name the<br />attack</>}
      deck={<>Pick a flow the model has never seen. {C.best} finds the {C.k} closest neighbours among {C.trainRows.toLocaleString('en-US')} training flows and lets them vote, closer ones counting more. On the full test set of {C.testRows.toLocaleString('en-US')} flows it is right {pct(best.acc, 1)} of the time.</>}
    >
      <div className="mb-6 flex flex-wrap gap-2" role="group" aria-label="Choose a flow by its true class">
        {C.order.map((c) => (
          <button key={c} type="button" onClick={() => pick(c)}
            className={cn('flex items-center gap-2 rounded-full border px-3.5 py-2 text-sm font-medium transition',
              flow.label === c ? 'border-ink text-ink' : 'border-hairline-strong text-ink-muted hover:border-ink-dim hover:text-ink')}>
            <span className="h-2 w-2 rounded-full" style={{ background: CLASS_COLOR[c] }} />{c}
          </button>
        ))}
        <button type="button" onClick={() => pick(null)}
          className="rounded-full bg-cyan px-4 py-2 text-sm font-semibold text-carbon transition hover:brightness-110">
          Random flow
        </button>
      </div>

      <div className="grid gap-px overflow-hidden rounded-2xl border border-hairline-strong bg-hairline-strong lg:grid-cols-[minmax(0,3fr)_minmax(0,5fr)_minmax(0,4fr)]">
        <div className="bg-panel p-6 md:p-8">
          <div className="mb-3 flex items-baseline justify-between">
            <span className="eyebrow">Flow #{flow.id}</span>
            <span className="text-xs text-ink-dim">from the test set</span>
          </div>
          <dl>
            <Fact k="Protocol" v={f.proto} />
            <Fact k="Service" v={f.service === '-' ? 'none' : f.service} />
            <Fact k="Connection state" v={f.state} />
            <Fact k="Duration" v={`${f.dur.toFixed(3)} s`} />
            <Fact k="Packets sent / back" v={`${f.spkts} / ${f.dpkts}`} />
            <Fact k="Bytes sent" v={bytes(f.sbytes)} />
            <Fact k="Bytes back" v={bytes(f.dbytes)} />
            <Fact k="TTL source / dest" v={`${f.sttl} / ${f.dttl}`} />
            <Fact k="Same-service connections" v={String(f.ct_srv_src)} />
          </dl>
        </div>

        <div className="relative grid place-items-center gap-6 overflow-hidden bg-[#03060a] p-6 md:p-10">
          <div className="pointer-events-none absolute inset-0 opacity-25 mix-blend-screen">
            <Video video="attackMap" cover />
          </div>
          <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_at_center,transparent_30%,#03060a_85%)]" />
          <div className="relative grid w-full justify-items-center gap-6">
            <Radar scanning={scanning} color={color} />
            <div className="grid justify-items-center gap-2 text-center" aria-live="polite">
              <span className="eyebrow">{scanning ? 'Scanning neighbours' : 'Model says'}</span>
              {/* keyed so React remounts the node instead of fighting the scramble over its text */}
              <div key={`${flow.id}-${scanned}`} ref={predRef} className="display text-[clamp(2rem,5vw,3.6rem)] leading-none" style={{ color, textShadow: `0 0 30px ${color}66` }}>
                {scanning ? '· · ·' : flow.pred}
              </div>
              {scanned && (
                <span className={cn('chip', correct ? 'border-cyan/40 text-cyan' : 'border-red/50 text-red')}>
                  {correct ? 'Correct' : `Wrong · really ${flow.label}`}
                </span>
              )}
            </div>
          </div>
        </div>

        <div className="grid content-start gap-6 bg-panel p-6 md:p-8">
          <div className="grid gap-3">
            <div className="flex items-baseline justify-between gap-4">
              <span className="eyebrow">Neighbour vote</span>
              <span className="text-xs text-ink-dim">distance-weighted</span>
            </div>
            {flow.proba.map((p) => (
              <div key={p.name} className="grid grid-cols-[7rem_minmax(0,1fr)_3rem] items-center gap-3 text-sm">
                <span className="text-ink-muted">{p.name}</span>
                <span className="relative h-2.5 overflow-hidden rounded-full bg-panel-2">
                  <span className="absolute inset-y-0 left-0 rounded-full transition-[width] duration-700 ease-out"
                    style={{ width: scanned ? `${p.p * 100}%` : '0%', background: CLASS_COLOR[p.name] }} />
                </span>
                <span className="num text-right">{pct(p.p)}</span>
              </div>
            ))}
          </div>
          <div className="grid gap-2 border-t border-hairline-strong pt-5">
            <span className="eyebrow">True class</span>
            <p className="font-medium" style={{ color: CLASS_COLOR[flow.label] }}>{flow.label}</p>
            <p className="text-sm leading-relaxed text-ink-muted">{CLASS_NOTE[flow.label]}</p>
          </div>
          <p className="text-xs leading-relaxed text-ink-dim">
            Each class has two flows the model got right and, where one exists, one it got wrong. Click a class again to see the next.
          </p>
        </div>
      </div>
    </Chapter>
  )
}
