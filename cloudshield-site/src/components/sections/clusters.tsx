import { useEffect, useRef, useState } from 'react'
import { Chapter } from '@/components/chapter'
import { gsap, ScrollTrigger, useGSAP, prefersReducedMotion } from '@/lib/gsap'
import { CLASS_COLOR, CLUSTER_COLOR, D, int, pct } from '@/lib/data'
import { cn } from '@/lib/utils'

const K = D.clustering
const ORDER = D.classification.order

type ColorBy = 'cluster' | 'class'

/**
 * Particle field. On scroll the flows fly from random noise to their place on the first two
 * principal components; colour switches between the K-Means cluster and the true class.
 */
function Field({ colorBy }: { colorBy: ColorBy }) {
  const wrap = useRef<HTMLDivElement>(null)
  const canvas = useRef<HTMLCanvasElement>(null)
  const state = useRef({ t: 0, mix: 0 })
  const draw = useRef<() => void>(() => {})

  useEffect(() => {
    const cv = canvas.current!
    const ctx = cv.getContext('2d')!
    const rand = K.points.map((_, i) => {
      const a = Math.sin(i * 12.9898) * 43758.5453
      const b = Math.sin(i * 78.233) * 12543.123
      return [a - Math.floor(a), b - Math.floor(b)]
    })
    const clusterCol = K.points.map((p) => CLUSTER_COLOR[p[2] % CLUSTER_COLOR.length])
    const classCol = K.points.map((p) => CLASS_COLOR[ORDER[p[3]]])
    const hex = (h: string) => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)]
    const cA = clusterCol.map(hex), cB = classCol.map(hex)

    const resize = () => {
      const r = cv.getBoundingClientRect()
      const dpr = Math.min(2, window.devicePixelRatio || 1)
      cv.width = r.width * dpr
      cv.height = r.height * dpr
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      draw.current()
    }
    draw.current = () => {
      const { width: w, height: h } = cv.getBoundingClientRect()
      const { t, mix } = state.current
      ctx.clearRect(0, 0, w, h)
      ctx.globalCompositeOperation = 'lighter'
      const pad = 24
      K.points.forEach((p, i) => {
        const x = pad + (rand[i][0] + (p[0] - rand[i][0]) * t) * (w - 2 * pad)
        const y = pad + (1 - (rand[i][1] + (p[1] - rand[i][1]) * t)) * (h - 2 * pad)
        const a = cA[i], b = cB[i]
        ctx.fillStyle = `rgba(${a[0] + (b[0] - a[0]) * mix},${a[1] + (b[1] - a[1]) * mix},${a[2] + (b[2] - a[2]) * mix},${0.25 + 0.45 * t})`
        ctx.beginPath()
        ctx.arc(x, y, 1.7, 0, Math.PI * 2)
        ctx.fill()
      })
      ctx.globalCompositeOperation = 'source-over'
    }
    resize()
    const ro = new ResizeObserver(resize)
    ro.observe(cv)
    return () => ro.disconnect()
  }, [])

  useGSAP(() => {
    if (prefersReducedMotion()) {
      state.current.t = 1
      draw.current()
      return
    }
    gsap.to(state.current, {
      t: 1, ease: 'power2.inOut', onUpdate: () => draw.current(),
      scrollTrigger: { trigger: wrap.current, start: 'top 85%', end: 'center 50%', scrub: 0.6 },
    })
    ScrollTrigger.refresh()
  }, { scope: wrap })

  useEffect(() => {
    gsap.to(state.current, { mix: colorBy === 'class' ? 1 : 0, duration: prefersReducedMotion() ? 0 : 0.8, ease: 'power2.inOut', onUpdate: () => draw.current() })
  }, [colorBy])

  return (
    <div ref={wrap} className="relative aspect-[4/3] w-full overflow-hidden rounded-xl bg-[#03060a]">
      <div className="grid-bg absolute inset-0 opacity-60" />
      <canvas ref={canvas} className="absolute inset-0 h-full w-full" role="img"
        aria-label={`${K.points.length} flows plotted on the first two principal components, coloured by ${colorBy}`} />
      <span className="num absolute bottom-3 left-4 text-[0.65rem] text-ink-dim">PC1 · {pct(K.pcaVariance[0])} of variance · axes compressed</span>
      <span className="num absolute left-3 top-4 origin-top-left translate-y-24 -rotate-90 text-[0.65rem] text-ink-dim">PC2 · {pct(K.pcaVariance[1])}</span>
    </div>
  )
}

function Elbow() {
  const w = 300, h = 140
  const e = K.elbow
  const x = (k: number) => 16 + ((k - e[0].k) / (e[e.length - 1].k - e[0].k)) * (w - 32)
  const imax = Math.max(...e.map((d) => d.inertia)), imin = Math.min(...e.map((d) => d.inertia))
  const y = (v: number) => 14 + (1 - (v - imin) / (imax - imin)) * (h - 40)
  const smax = Math.max(...e.map((d) => d.silhouette)), smin = Math.min(...e.map((d) => d.silhouette))
  const ys = (v: number) => 14 + (1 - (v - smin) / (smax - smin || 1)) * (h - 40)
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full" role="img" aria-label={`Elbow curve; the bend is at k = ${K.k}`}>
      <path d={e.map((d, i) => `${i ? 'L' : 'M'}${x(d.k)} ${ys(d.silhouette)}`).join('')} fill="none" stroke="#ffb547" strokeOpacity="0.6" strokeDasharray="3 3" />
      <path d={e.map((d, i) => `${i ? 'L' : 'M'}${x(d.k)} ${y(d.inertia)}`).join('')} fill="none" stroke="#36e2d4" strokeWidth="1.8" />
      {e.map((d) => (
        <g key={d.k}>
          <circle cx={x(d.k)} cy={y(d.inertia)} r={d.k === K.k ? 5 : 2.5} fill={d.k === K.k ? '#36e2d4' : '#05080c'} stroke="#36e2d4" />
          <text x={x(d.k)} y={h - 6} fontSize="9" textAnchor="middle" fill={d.k === K.k ? '#eaf2f5' : 'rgba(234,242,245,0.45)'} fontFamily="JetBrains Mono">{d.k}</text>
        </g>
      ))}
    </svg>
  )
}

export function Clusters() {
  const [colorBy, setColorBy] = useState<ColorBy>('cluster')
  const km = K.kmeans, ag = K.agglomerative

  return (
    <Chapter
      id="clusters"
      index="07"
      eyebrow="Clustering · no labels"
      title={<>Let the traffic<br />sort itself</>}
      deck={<>Hide the labels and ask K-Means to group {int(K.sample)} flows by shape alone. The elbow of the curve says {K.k} groups. Then switch the colours to the true classes and see how much the groups line up with attacks.</>}
    >
      <div className="grid gap-4 lg:grid-cols-[minmax(0,7fr)_minmax(0,5fr)]">
        <div className="grid content-start gap-4 rounded-2xl border border-hairline-strong bg-panel p-5 md:p-6">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <span className="eyebrow">Flows on two principal components</span>
            <div className="flex gap-1 rounded-full border border-hairline-strong p-1" role="tablist">
              {([['cluster', 'K-Means cluster'], ['class', 'True class']] as const).map(([k, l]) => (
                <button key={k} type="button" role="tab" aria-selected={colorBy === k} onClick={() => setColorBy(k)}
                  className={cn('rounded-full px-3 py-1.5 text-xs font-medium transition', colorBy === k ? 'bg-cyan text-carbon' : 'text-ink-muted hover:text-ink')}>{l}</button>
              ))}
            </div>
          </div>
          <Field colorBy={colorBy} />
          <div className="flex flex-wrap gap-x-4 gap-y-1.5 text-xs text-ink-muted">
            {colorBy === 'cluster'
              ? K.clusters.map((c) => <span key={c.id} className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-full" style={{ background: CLUSTER_COLOR[c.id] }} />Cluster {c.id + 1}</span>)
              : ORDER.map((n) => <span key={n} className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-full" style={{ background: CLASS_COLOR[n] }} />{n}</span>)}
          </div>
        </div>

        <div className="grid content-start gap-4">
          <div className="grid gap-3 rounded-2xl border border-hairline-strong bg-panel p-6">
            <div className="flex items-baseline justify-between gap-4">
              <span className="eyebrow">Choosing k</span>
              <span className="text-xs text-ink-dim"><span className="text-cyan">inertia</span> · <span className="text-amber">silhouette</span></span>
            </div>
            <Elbow />
            <div className="grid grid-cols-3 gap-px overflow-hidden rounded-lg bg-hairline-strong text-center text-xs">
              {[['', 'K-Means', 'Agglomerative'], ['Silhouette ↑', km.silhouette.toFixed(3), ag.silhouette.toFixed(3)],
                ['Davies-Bouldin ↓', km.daviesBouldin.toFixed(2), ag.daviesBouldin.toFixed(2)], ['Calinski-Harabasz ↑', int(km.calinskiHarabasz), int(ag.calinskiHarabasz)]].map((row, i) => (
                row.map((cell, j) => (
                  <span key={`${i}-${j}`} className={cn('bg-panel-2 px-2 py-2', i === 0 || j === 0 ? 'text-ink-dim' : 'num text-ink', j === 0 && 'text-left')}>{cell}</span>
                ))
              ))}
            </div>
          </div>
          <div className="grid gap-3 rounded-2xl border border-hairline-strong bg-panel p-6">
            <span className="eyebrow">What is inside each cluster</span>
            {K.clusters.map((c) => (
              <div key={c.id} className="grid gap-1.5">
                <div className="flex items-baseline justify-between text-sm">
                  <span className="flex items-center gap-2"><span className="h-2 w-2 rounded-full" style={{ background: CLUSTER_COLOR[c.id] }} />Cluster {c.id + 1}</span>
                  <span className="num text-xs text-ink-dim">{int(c.size)} flows</span>
                </div>
                <div className="flex h-2.5 overflow-hidden rounded-full bg-panel-2" title={c.mix.map((m) => `${m.name} ${pct(m.share)}`).join(', ')}>
                  {c.mix.map((m) => <span key={m.name} style={{ width: `${m.share * 100}%`, background: CLASS_COLOR[m.name] }} />)}
                </div>
                <span className="text-xs text-ink-muted">{c.mix.slice(0, 3).map((m) => `${m.name} ${pct(m.share)}`).join(' · ')}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
      <p className="mt-6 max-w-[80ch] text-xs leading-relaxed text-ink-dim">{K.note}</p>
    </Chapter>
  )
}
