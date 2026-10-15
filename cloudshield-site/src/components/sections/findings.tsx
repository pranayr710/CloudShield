import { useRef } from 'react'
import { Chapter } from '@/components/chapter'
import { Video } from '@/components/video'
import { gsap, SplitText, useGSAP, prefersReducedMotion } from '@/lib/gsap'
import { D, T, int, pct, tonLabel } from '@/lib/data'

const C = T.classification
const K = T.clustering
const by = (key: string) => C.models.find((m) => m.key === key)!
const base = by(C.bestBase), tuned = by(C.best), gnb = by('Gaussian NB')
const strat = C.cv5.find((c) => c.model === C.best && c.kind === 'stratified')
const worst = [...C.order].sort((a, b) => C.recall[a] - C.recall[b])[0]
const wRow = C.confusion.norm[C.confusion.rows.indexOf(worst)]
const worstTo = C.confusion.cols.filter((c) => c !== worst).reduce((b, c) => (wRow[C.confusion.cols.indexOf(c)] > wRow[C.confusion.cols.indexOf(b)] ? c : b))
const normalCluster = [...K.clusters].filter((c) => c.size >= 50)
  .map((c) => ({ c, n: c.mix.find((m) => m.name === 'normal')?.share ?? 0 })).sort((a, b) => b.n - a.n)[0]

const FINDINGS: [string, string, string][] = [
  [C.final.f1w.toFixed(3), 'Attacks can be named on a new network',
    `Tested only on ${C.testMachines} machines it never saw in training, the ${C.bestDisplay} model labels ${int(C.testRows)} flows with weighted F1 ${C.final.f1w.toFixed(4)} and accuracy ${pct(C.final.acc, 1)}.`],
  [`+${(tuned.f1w - base.f1w).toFixed(3)}`, 'Tuning paid off',
    `A grid search scored with grouped cross-validation on the training machines moved ${base.model} from weighted F1 ${base.f1w.toFixed(3)} to ${tuned.f1w.toFixed(3)} (${C.finalParamsText}). Gaussian NB, which assumes independent features, manages ${gnb.f1w.toFixed(3)}.`],
  [pct(C.recall[worst], 1), `${tonLabel(worst)} is the weak spot`,
    `Only ${pct(C.recall[worst], 1)} of ${tonLabel(worst)} flows from the unseen machines are recognised; the rest are mostly labelled ${tonLabel(worstTo)}.`],
  [strat ? `${strat.mean.toFixed(2)} → ${C.final.f1w.toFixed(2)}` : '–', 'Cross-validation flatters',
    `Ordinary stratified 5-fold cross-validation, where folds share machines, scores ${strat?.mean.toFixed(3)}; on unseen machines the same model scores ${C.final.f1w.toFixed(3)}. Only a test set grouped by machine shows the real gap.`],
  [String(K.k), 'Behaviour groups without labels',
    `K-Means finds ${K.k} clusters that are stable across resamples (ARI ${K.stability.ari?.[0].toFixed(2)}). One is ${pct(normalCluster.n)} normal traffic; the others mix attack families.`],
  [D.regression.models[0].r2.toFixed(4), 'Reply size is predictable',
    `On ${D.overview.dataset.name}, a ${D.regression.best} forecasts the bytes a flow sends back with R² ${D.regression.models[0].r2.toFixed(4)} and a typical error of ${int(D.regression.models[0].mae)} bytes.`],
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
    <section ref={root} className="screen relative grid h-[100svh] place-items-center overflow-hidden" aria-label="Closing">
      <div data-video className="absolute inset-0">
        <Video video="shield" cover />
      </div>
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,rgba(21,19,15,0.25)_0%,rgba(21,19,15,0.9)_75%)]" />
      <div className="relative grid justify-items-center gap-6 px-6 text-center">
        <span className="eyebrow">Thank you</span>
        <h2 ref={title} className="display text-[clamp(2.6rem,10vw,8.5rem)] leading-none text-ink">
          Cloud<span className="text-red">Shield</span>
        </h2>
        <p className="max-w-[40ch] font-serif text-[clamp(1.3rem,2.4vw,1.9rem)] italic text-ink-muted">Questions are welcome.</p>
      </div>
    </section>
  )
}

export function Findings() {
  return (
    <>
      <Chapter id="findings" index="11" eyebrow="What the data said" title={<>Six findings,<br />three tracks</>}>
        <ol className="grid gap-px overflow-hidden rounded-2xl border border-hairline-strong bg-hairline-strong md:grid-cols-2 lg:grid-cols-3">
          {FINDINGS.map(([value, head, body], i) => (
            <li key={head} className="grain group grid content-start gap-4 bg-panel p-6 transition-colors hover:bg-paper md:p-8">
              <div className="flex items-baseline justify-between gap-4">
                <span className="font-serif text-3xl italic text-red">{i + 1}</span>
                <span className="num text-right text-[clamp(1.3rem,2.2vw,1.9rem)] font-medium">{value}</span>
              </div>
              <h3 className="text-lg font-semibold">{head}</h3>
              <p className="text-sm leading-relaxed text-ink-muted">{body}</p>
            </li>
          ))}
        </ol>
      </Chapter>
      <Finale />
    </>
  )
}
