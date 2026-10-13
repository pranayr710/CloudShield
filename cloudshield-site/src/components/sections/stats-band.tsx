import { useRef } from 'react'
import { gsap, useGSAP, prefersReducedMotion } from '@/lib/gsap'
import { D, T, int } from '@/lib/data'

const TRACKS = [
  {
    no: 'I', track: 'Classification', dataset: T.overview.name, question: 'Which attack is this flow?',
    value: T.classification.final.f1w.toFixed(3), unit: 'weighted F1 on unseen machines', href: '#classifier',
  },
  {
    no: 'II', track: 'Clustering', dataset: T.overview.name, question: 'What kinds of traffic exist, with no labels?',
    value: String(T.clustering.k), unit: 'behaviour clusters', href: '#clusters',
  },
  {
    no: 'III', track: 'Regression', dataset: D.overview.dataset.name, question: 'How many bytes will the reply carry?',
    value: D.regression.models[0].r2.toFixed(4), unit: 'R² on unseen flows', href: '#forecaster',
  },
]

const STATS = [
  [int(T.overview.rows), 'TON_IoT flows'],
  [String(T.overview.classes.length), 'Traffic classes'],
  [String(T.overview.machines), 'Source machines'],
  ['10', 'Classifiers'],
  [int(D.overview.rows), 'UNSW-NB15 flows'],
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
      gsap.from('[data-track]', {
        y: 40, opacity: 0, rotate: (i) => [-1.5, 1, -0.5][i], stagger: 0.12, duration: 1, ease: 'expo.out',
        scrollTrigger: { trigger: '[data-track]', start: 'top 85%', once: true },
      })
    },
    { scope: root },
  )

  const intro = 'Every connection on a network leaves a record: how long it lasted, how many packets went each way, which service it used. CloudShield asks three questions of those records, each with the method that suits it.'

  return (
    <section ref={root} className="gutter relative border-b border-hairline-strong py-20 md:py-28">
      <div className="mx-auto grid max-w-[1360px] gap-14">
        <p data-intro className="max-w-[34ch] font-serif text-[clamp(1.8rem,3.6vw,3.2rem)] italic leading-[1.15] text-ink">
          {intro.split(' ').map((w, i) => <span key={i} className="word">{w} </span>)}
        </p>

        <div className="grid gap-4 md:grid-cols-3">
          {TRACKS.map((t) => (
            <a key={t.no} href={t.href} data-track
              className="grain group grid content-between gap-8 rounded-2xl border border-hairline-strong bg-panel p-6 shadow-[0_1px_0_rgba(22,20,15,0.08)] transition hover:-translate-y-1 hover:shadow-[0_18px_40px_rgba(22,20,15,0.12)] md:p-8">
              <div className="flex items-start justify-between gap-4">
                <div className="grid gap-1">
                  <span className="eyebrow">Track {t.no} · {t.dataset}</span>
                  <h3 className="display text-[1.5rem]">{t.track}</h3>
                </div>
                <span className="font-serif text-4xl italic text-red">{t.no}</span>
              </div>
              <p className="font-serif text-[1.35rem] italic leading-snug text-ink-muted">{t.question}</p>
              <div className="flex items-end justify-between gap-4 border-t border-hairline-strong pt-4">
                <span className="num text-[2.2rem] font-medium leading-none">{t.value}</span>
                <span className="text-right text-xs text-ink-dim">{t.unit}<br /><span className="text-red transition group-hover:translate-x-1">read →</span></span>
              </div>
            </a>
          ))}
        </div>

        <dl className="grid grid-cols-2 gap-px overflow-hidden rounded-xl border border-hairline-strong bg-hairline-strong sm:grid-cols-3 lg:grid-cols-5">
          {STATS.map(([value, label]) => (
            <div key={label} className="grid gap-2 bg-paper p-5 md:p-7">
              <dt className="eyebrow order-2">{label}</dt>
              <dd className="num order-1 text-[clamp(1.8rem,3.4vw,2.8rem)] font-medium" data-scramble={value}>{value}</dd>
            </div>
          ))}
        </dl>
      </div>
    </section>
  )
}
