import { useRef } from 'react'
import { gsap, ScrollTrigger, useGSAP, prefersReducedMotion } from '@/lib/gsap'
import { D, int, pct } from '@/lib/data'

const { overview: O, classification: C, regression: R } = D
const bestCls = C.models.find((m) => m.model === C.best) ?? C.models[0]
const bestReg = R.models[0]

const ITEMS: [string, string, 'cyan' | 'red'][] = [
  ['Network flows', int(O.rows), 'cyan'],
  ['Attack share', pct(O.attackShare, 1), 'red'],
  ['Duplicates removed', pct(O.duplicateShare, 1), 'cyan'],
  ['Class imbalance', `${O.imbalance} : 1`, 'red'],
  [`${C.best} accuracy`, pct(bestCls.acc, 1), 'cyan'],
  ['Normal traffic recall', pct(C.perClass.find((p) => p.name === 'Normal')!.recall, 1), 'cyan'],
  ['Worms recall', pct(C.perClass.find((p) => p.name === 'Worms')!.recall, 1), 'red'],
  [`${bestReg.model} R²`, bestReg.r2.toFixed(4), 'cyan'],
  ['Typical byte error', `${int(bestReg.mae)} B`, 'cyan'],
]

/** A SOC-screen ticker of findings. Its speed and lean follow how fast the page is scrolled. */
export function Ticker() {
  const root = useRef<HTMLDivElement>(null)
  const track = useRef<HTMLDivElement>(null)

  useGSAP(
    () => {
      if (prefersReducedMotion()) return
      const loop = gsap.to(track.current, { xPercent: -50, duration: 48, ease: 'none', repeat: -1 })
      const lean = gsap.quickTo(track.current, 'skewX', { duration: 0.5, ease: 'power3.out' })
      ScrollTrigger.create({
        trigger: root.current,
        start: 'top bottom',
        end: 'bottom top',
        onUpdate: (self) => {
          const v = self.getVelocity()
          loop.timeScale(1 + Math.min(6, Math.abs(v) / 400))
          lean(gsap.utils.clamp(-10, 10, -v / 250))
          gsap.to(loop, { timeScale: 1, duration: 1.2, overwrite: true, delay: 0.1 })
        },
      })
    },
    { scope: root },
  )

  const row = ITEMS.map(([k, v, tone]) => (
    <span key={k} className="flex shrink-0 items-center gap-4 pr-12">
      <span className={`h-1.5 w-1.5 rounded-full ${tone === 'red' ? 'bg-red shadow-[0_0_10px_#ff3b5c]' : 'bg-cyan shadow-[0_0_10px_#36e2d4]'}`} />
      <span className="eyebrow text-ink-muted">{k}</span>
      <span className="num text-[1.3rem] font-medium text-ink">{v}</span>
    </span>
  ))

  return (
    <div ref={root} className="overflow-hidden border-y border-hairline-strong bg-panel py-5" aria-label="Key findings">
      <div ref={track} className="flex w-max will-change-transform">
        <div className="flex">{row}</div>
        <div className="flex" aria-hidden>{row}</div>
      </div>
    </div>
  )
}
