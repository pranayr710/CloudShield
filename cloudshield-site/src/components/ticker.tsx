import { useRef } from 'react'
import { gsap, ScrollTrigger, useGSAP, prefersReducedMotion } from '@/lib/gsap'
import { D, T, int, pct } from '@/lib/data'

const C = T.classification

const ITEMS: [string, string][] = [
  ['TON_IoT flows', int(T.overview.rows)],
  ['Attack share', pct(T.overview.attackShare, 1)],
  ['Classifiers compared', '10'],
  ['Final model', C.bestDisplay],
  ['Weighted F1, unseen machines', C.final.f1w.toFixed(4)],
  ['Accuracy, unseen machines', pct(C.final.acc, 1)],
  ['Unseen test machines', String(C.testMachines)],
  ['Traffic clusters', String(T.clustering.k)],
  [`Regression R² (${D.overview.dataset.name})`, D.regression.models[0].r2.toFixed(4)],
]

/** A news-wire ticker of findings. Its speed and lean follow how fast the page is scrolled. */
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

  const row = ITEMS.map(([k, v]) => (
    <span key={k} className="flex shrink-0 items-center gap-4 pr-12">
      <span className="h-2 w-2 rotate-45 bg-red" />
      <span className="eyebrow">{k}</span>
      <span className="num text-[1.3rem] font-medium text-ink">{v}</span>
    </span>
  ))

  return (
    <div ref={root} className="screen overflow-hidden py-5" aria-label="Key findings">
      <div ref={track} className="flex w-max will-change-transform">
        <div className="flex">{row}</div>
        <div className="flex" aria-hidden>{row}</div>
      </div>
    </div>
  )
}
