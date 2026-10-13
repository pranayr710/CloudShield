import { useRef } from 'react'
import { HeroScrub } from '@/components/ui/hero-scrub'
import { gsap, useGSAP, prefersReducedMotion } from '@/lib/gsap'
import { D, int, pct } from '@/lib/data'

const O = D.overview
const FRAMES = 140
const frameUrl = (i: number) => `/frames/server/${String(i + 1).padStart(4, '0')}.webp`

const BOOT: [string, string][] = [
  [`loading ${O.dataset.name}`, `${int(O.rows)} flows`],
  ['auditing exact duplicates', int(O.duplicates)],
  ['fitting scaler on train only', 'ok'],
  ['K-Nearest Neighbors', 'armed'],
  ['Random Forest', 'armed'],
]

/** Terminal boot sequence that lifts away, then the scroll-scrubbed data-centre hero. */
export function Hero() {
  const curtain = useRef<HTMLDivElement>(null)

  useGSAP(() => {
    if (prefersReducedMotion()) {
      gsap.set(curtain.current, { display: 'none' })
      return
    }
    const lines = curtain.current!.querySelectorAll<HTMLElement>('[data-line]')
    const values = curtain.current!.querySelectorAll<HTMLElement>('[data-value]')
    const granted = curtain.current!.querySelector<HTMLElement>('[data-granted]')!
    const bar = curtain.current!.querySelector<HTMLElement>('[data-bar]')!
    gsap.set(lines, { opacity: 0 })
    gsap.set(granted, { opacity: 0 })
    const tl = gsap.timeline({ delay: 0.35 })
    lines.forEach((l, i) => {
      tl.fromTo(l, { opacity: 0, x: -8 }, { opacity: 1, x: 0, duration: 0.2 }, i * 0.32)
      tl.fromTo(values[i], { opacity: 0 }, {
        opacity: 1,
        scrambleText: { text: values[i].dataset.value!, chars: '0123456789#%&', speed: 0.9 },
        duration: 0.35, ease: 'none',
      }, '<0.12')
    })
    tl.fromTo(bar, { scaleX: 0 }, { scaleX: 1, duration: lines.length * 0.32, ease: 'none' }, 0)
    tl.to(granted, { opacity: 1, scrambleText: { text: 'Access granted', chars: 'upperCase', speed: 0.8 }, duration: 0.5 }, '+=0.15')
    tl.to(curtain.current, { yPercent: -100, duration: 1.1, ease: 'expo.inOut' }, '+=0.45')
    tl.set(curtain.current, { display: 'none' })
    curtain.current!.addEventListener('click', () => tl.progress(1), { once: true })
  })

  return (
    <div id="top">
      <h1 className="sr-only">CloudShield: machine learning on network traffic</h1>
      <HeroScrub
        frameCount={FRAMES}
        frameUrl={frameUrl}
        titleTop="Cloud"
        titleBottom="Shield"
        accentHex="#05080c"
        immersedLines={[`${int(O.rows)} network flows.`, `${pct(O.attackShare)} of them are attacks.`, 'Can a model tell which?']}
        kicker={
          <div className="grid justify-items-center gap-3 text-center">
            <span className="eyebrow">Machine learning on network traffic · {O.dataset.name}</span>
            <span className="flex items-center gap-2 text-xs text-ink-dim">
              <span className="inline-block h-6 w-px animate-pulse bg-cyan" /> Scroll
            </span>
          </div>
        }
      />

      <div ref={curtain} className="grid-bg fixed inset-0 z-[70] grid cursor-pointer place-items-center bg-carbon px-6" role="presentation">
        <div className="scanlines relative w-full max-w-[640px] overflow-hidden rounded-xl border border-hairline-strong bg-panel/90 p-6 md:p-8">
          <div className="mb-6 flex items-center gap-2" aria-hidden>
            <span className="h-2.5 w-2.5 rounded-full bg-red/80" />
            <span className="h-2.5 w-2.5 rounded-full bg-amber/80" />
            <span className="h-2.5 w-2.5 rounded-full bg-cyan/80" />
            <span className="num ml-3 text-xs text-ink-dim">cloudshield --boot</span>
          </div>
          <ul className="num grid gap-2.5 text-[0.82rem] md:text-sm">
            {BOOT.map(([label, value]) => (
              <li key={label} data-line className="flex items-baseline gap-3">
                <span className="text-cyan">›</span>
                <span data-label className="text-ink-muted">{label}</span>
                <span className="min-w-6 flex-1 translate-y-[-3px] border-b border-dotted border-hairline-strong" />
                <span data-value={value} className="text-ink">{value}</span>
              </li>
            ))}
          </ul>
          <div className="mt-6 h-px bg-hairline-strong">
            <div data-bar className="h-px origin-left bg-cyan shadow-[0_0_12px_#36e2d4]" />
          </div>
          <p data-granted className="display mt-6 text-[clamp(1.4rem,4vw,2rem)] text-cyan glow-cyan">Access granted</p>
          <p className="mt-4 text-xs text-ink-dim">Click to skip</p>
        </div>
      </div>
    </div>
  )
}
