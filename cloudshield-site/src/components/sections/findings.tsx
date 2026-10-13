import { useRef } from 'react'
import { Chapter } from '@/components/chapter'
import { Video } from '@/components/video'
import { gsap, SplitText, useGSAP, prefersReducedMotion } from '@/lib/gsap'
import { D, int, pct } from '@/lib/data'

const { classification: C, regression: R, clustering: K } = D
const recall = (n: string) => C.perClass.find((p) => p.name === n)!.recall
const poly = R.models.find((m) => m.model.startsWith('Polynomial'))!
const generic = K.clusters.map((c) => ({ c, g: c.mix.find((m) => m.name === 'Generic')?.share ?? 0 })).sort((a, b) => b.g - a.g)[0]

const FINDINGS: [string, string, string][] = [
  [R.models[0].r2.toFixed(4), 'Response size is predictable', `A ${R.best} forecasts the bytes a flow sends back with a typical error of ${int(R.models[0].mae)} bytes, without seeing any column that is a formula of the answer.`],
  [`${pct(recall('Normal'))} vs ${pct(recall('Worms'))}`, 'Common is easy, rare is hard', `Normal traffic is recognised ${pct(recall('Normal'), 1)} of the time, Worms ${pct(recall('Worms'), 1)}. With ${D.overview.imbalance} Normal flows per Worm, there is little to learn from.`],
  ['5 of 5', 'Every model fails the same way', 'All five classifiers, built on very different ideas, confuse the same rare families with Exploits and DoS. That is a property of the flow statistics, not of any one algorithm.'],
  [`${poly.r2Val.toFixed(3)} → ${poly.r2.toFixed(3)}`, 'Only a holdout catches overfitting', 'The degree-2 polynomial looked strong on validation and dropped on the untouched test set. Without that last check we would have trusted it.'],
  [pct(generic.g), 'Some attacks have one signature', `Without labels, K-Means put Generic attacks into a cluster of their own (${pct(generic.g)} pure). Generic flows are near-identical, which is also why ${pct(D.overview.duplicateShare)} of the data is duplicates.`],
]

function Finale() {
  const root = useRef<HTMLElement>(null)
  const title = useRef<HTMLHeadingElement>(null)

  useGSAP(() => {
    if (prefersReducedMotion()) return
    const split = SplitText.create(title.current!, { type: 'chars' })
    gsap.from(split.chars, {
      yPercent: 120, opacity: 0, stagger: 0.04, duration: 1, ease: 'expo.out',
      scrollTrigger: { trigger: root.current, start: 'top 60%', once: true },
    })
    gsap.fromTo(root.current!.querySelector('[data-video]'), { scale: 1.2 }, {
      scale: 1, ease: 'none', scrollTrigger: { trigger: root.current, start: 'top bottom', end: 'bottom bottom', scrub: true },
    })
    return () => split.revert()
  }, { scope: root })

  return (
    <section ref={root} className="relative grid h-[100svh] place-items-center overflow-hidden" aria-label="Closing">
      <div data-video className="absolute inset-0">
        <Video video="shield" cover />
      </div>
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,rgba(5,8,12,0.25)_0%,rgba(5,8,12,0.9)_75%)]" />
      <div className="relative grid justify-items-center gap-6 px-6 text-center">
        <span className="eyebrow">Thank you</span>
        <h2 ref={title} className="display text-[clamp(2.6rem,10vw,8.5rem)] leading-none">
          Cloud<span className="text-cyan glow-cyan">Shield</span>
        </h2>
        <p className="max-w-[40ch] font-serif text-[clamp(1.3rem,2.4vw,1.9rem)] italic text-ink-muted">Questions are welcome.</p>
      </div>
    </section>
  )
}

export function Findings() {
  return (
    <>
      <Chapter id="findings" index="08" eyebrow="What we learned" title={<>Five things<br />the data said</>}>
        <ol className="grid gap-px overflow-hidden rounded-2xl border border-hairline-strong bg-hairline-strong md:grid-cols-2 lg:grid-cols-3">
          {FINDINGS.map(([value, head, body], i) => (
            <li key={head} className="group grid content-start gap-4 bg-panel p-6 transition-colors hover:bg-panel-2 md:p-8">
              <div className="flex items-baseline justify-between gap-4">
                <span className="num text-sm text-ink-dim">0{i + 1}</span>
                <span className="num text-right text-[clamp(1.4rem,2.4vw,2rem)] font-medium text-cyan transition group-hover:[text-shadow:0_0_28px_rgba(54,226,212,0.45)]">{value}</span>
              </div>
              <h3 className="text-lg font-semibold">{head}</h3>
              <p className="text-sm leading-relaxed text-ink-muted">{body}</p>
            </li>
          ))}
          <li className="relative grid min-h-[220px] content-end overflow-hidden bg-[#03060a] p-6 md:p-8">
            <Video video="fibre" cover className="opacity-70" />
            <span className="relative eyebrow text-ink">Next: the Review 2 dataset</span>
          </li>
        </ol>
      </Chapter>
      <Finale />
    </>
  )
}
