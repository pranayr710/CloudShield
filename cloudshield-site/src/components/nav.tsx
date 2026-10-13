import { useRef } from 'react'
import { gsap, useGSAP } from '@/lib/gsap'

const LINKS = [
  ['dataset', 'Dataset'],
  ['classifier', 'Classifier'],
  ['confusion', 'Errors'],
  ['forecaster', 'Forecaster'],
  ['models', 'Models'],
  ['traps', 'Traps'],
  ['clusters', 'Clusters'],
  ['findings', 'Findings'],
] as const

export function Nav() {
  const bar = useRef<HTMLDivElement>(null)

  useGSAP(() => {
    gsap.to(bar.current, { scaleX: 1, ease: 'none', scrollTrigger: { start: 0, end: 'max', scrub: 0.3 } })
  })

  return (
    <nav
      aria-label="Sections"
      className="fixed inset-x-0 top-0 z-50 border-b border-hairline bg-carbon/70 backdrop-blur-md"
      style={{ paddingTop: 'env(safe-area-inset-top, 0px)' }}
    >
      <div className="gutter mx-auto flex h-14 max-w-[1560px] items-center gap-6">
        <a href="#top" className="flex shrink-0 items-center gap-2.5" aria-label="CloudShield, back to top">
          <svg viewBox="0 0 32 32" className="h-6 w-6" aria-hidden>
            <path d="M16 5 7 8.5v6.8c0 5.6 3.8 10 9 11.7 5.2-1.7 9-6.1 9-11.7V8.5z" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinejoin="round" className="text-cyan" />
            <circle cx="16" cy="14.5" r="2.6" className="fill-cyan" />
          </svg>
          <span className="display text-[0.85rem] tracking-normal">Cloud<span className="text-cyan">Shield</span></span>
        </a>
        <div className="-mr-2 hidden flex-1 justify-end gap-1 md:flex">
          {LINKS.map(([id, label]) => (
            <a key={id} href={`#${id}`}
              className="shrink-0 rounded-full px-3 py-1.5 text-[0.82rem] font-medium text-ink-muted transition-colors hover:bg-panel-2 hover:text-ink">
              {label}
            </a>
          ))}
        </div>
      </div>
      <div ref={bar} aria-hidden className="h-[2px] origin-left scale-x-0 bg-gradient-to-r from-cyan via-cyan to-red" />
    </nav>
  )
}
