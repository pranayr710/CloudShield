import { useRef } from 'react'
import { gsap, useGSAP, prefersReducedMotion } from '@/lib/gsap'
import { D, int } from '@/lib/data'

const O = D.overview

const STATS = [
  [int(O.rows), 'Network flows'],
  [String(O.features), 'Features per flow'],
  [String(O.classes - 1), 'Attack families'],
  [String(O.protocols), 'Protocols'],
  [int(O.duplicates), 'Exact duplicates'],
] as const

export function StatsBand() {
  const root = useRef<HTMLElement>(null)

  useGSAP(
    () => {
      if (prefersReducedMotion()) return
      root.current!.querySelectorAll<HTMLElement>('[data-scramble]').forEach((el, i) => {
        gsap.from(el, {
          scrambleText: { text: el.dataset.scramble!, chars: '0123456789', speed: 0.6 },
          duration: 1.4, delay: i * 0.08, ease: 'none',
          scrollTrigger: { trigger: root.current, start: 'top 80%', once: true },
        })
      })
      gsap.from('[data-intro] .word', {
        opacity: 0.12, stagger: 0.03, ease: 'none',
        scrollTrigger: { trigger: '[data-intro]', start: 'top 80%', end: 'bottom 55%', scrub: true },
      })
    },
    { scope: root },
  )

  const intro = `Every connection on a network leaves a record: how long it lasted, how many packets went each way, which service it used. We took ${int(O.rows)} of those records and asked three questions. Is this flow an attack? How much data will it send back? And what kinds of traffic are there at all?`

  return (
    <section ref={root} className="gutter relative border-y border-hairline py-20 md:py-28">
      <div className="mx-auto grid max-w-[1360px] gap-16">
        <p data-intro className="max-w-[34ch] font-serif text-[clamp(1.8rem,3.6vw,3.2rem)] italic leading-[1.15] text-ink">
          {intro.split(' ').map((w, i) => <span key={i} className="word">{w} </span>)}
        </p>
        <dl className="grid grid-cols-2 gap-px overflow-hidden rounded-xl border border-hairline-strong bg-hairline-strong sm:grid-cols-3 lg:grid-cols-5">
          {STATS.map(([value, label]) => (
            <div key={label} className="grid gap-2 bg-carbon p-5 md:p-7">
              <dt className="eyebrow order-2">{label}</dt>
              <dd className="num order-1 text-[clamp(1.8rem,3.4vw,2.8rem)] font-medium" data-scramble={value}>{value}</dd>
            </div>
          ))}
        </dl>
      </div>
    </section>
  )
}
