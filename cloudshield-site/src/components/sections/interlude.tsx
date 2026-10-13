import { useRef, type ReactNode } from 'react'
import { Video, type VideoKey } from '@/components/video'
import { gsap, useGSAP, prefersReducedMotion } from '@/lib/gsap'
import { cn } from '@/lib/utils'

type Props = {
  video: VideoKey
  quote: ReactNode
  value: string
  caption: string
  tone?: 'signal' | 'red'
}

/** Full-bleed video break between chapters, carrying one finding. */
export function Interlude({ video, quote, value, caption, tone = 'signal' }: Props) {
  const root = useRef<HTMLElement>(null)
  const frame = useRef<HTMLDivElement>(null)
  const quoteRef = useRef<HTMLParagraphElement>(null)
  const valueRef = useRef<HTMLSpanElement>(null)

  useGSAP(
    () => {
      if (prefersReducedMotion()) return
      gsap.fromTo(frame.current, { clipPath: 'inset(12% 10% 12% 10% round 24px)' }, {
        clipPath: 'inset(0% 0% 0% 0% round 16px)', ease: 'none',
        scrollTrigger: { trigger: root.current, start: 'top 85%', end: 'center 55%', scrub: true },
      })
      gsap.fromTo(quoteRef.current, { yPercent: 40, opacity: 0.2 }, {
        yPercent: -10, opacity: 1, ease: 'none',
        scrollTrigger: { trigger: root.current, start: 'top bottom', end: 'top 30%', scrub: true },
      })
      gsap.from(valueRef.current, {
        scrambleText: { text: value, chars: '0123456789%.', speed: 0.6 }, duration: 1.2, ease: 'none',
        scrollTrigger: { trigger: valueRef.current, start: 'top 85%', once: true },
      })
    },
    { scope: root },
  )

  return (
    <section ref={root} className="gutter py-16 md:py-24" aria-label="Interlude">
      <div className="mx-auto grid max-w-[1560px] gap-10">
        <div className="grid gap-6 md:grid-cols-[minmax(0,1fr)_auto] md:items-end">
          <p ref={quoteRef} className="max-w-[24ch] font-serif text-[clamp(2rem,4.4vw,3.8rem)] italic leading-[1.08]">{quote}</p>
          <div className="grid gap-1 md:text-right">
            <span ref={valueRef} className={cn('num text-[clamp(2.6rem,6vw,5rem)] font-medium leading-none', tone === 'red' ? 'text-red' : 'text-signal')}>{value}</span>
            <span className="eyebrow max-w-[34ch] md:justify-self-end">{caption}</span>
          </div>
        </div>
        <div ref={frame} className="overflow-hidden rounded-2xl bg-black">
          <Video video={video} />
        </div>
      </div>
    </section>
  )
}
