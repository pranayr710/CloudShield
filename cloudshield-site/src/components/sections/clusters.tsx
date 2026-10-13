import { useEffect, useRef, useState } from 'react'
import { Chapter } from '@/components/chapter'
import { gsap, ScrollTrigger, useGSAP, prefersReducedMotion } from '@/lib/gsap'
import { CLUSTER_COLOR, T, TON_COLOR, int, pct, tonLabel } from '@/lib/data'
import { cn } from '@/lib/utils'

const K = T.clustering
const ORDER = T.classification.order

type ColorBy = 'cluster' | 'class'

/** Short names for each cluster, from its two biggest traffic types. */
const NAME = (c: (typeof K.clusters)[number]) =>
  c.size < 50 ? 'outliers' : c.mix.slice(0, 2).map((m) => tonLabel(m.name)).join(' + ')

/**
 * Particle map. On scroll the flows fly from random noise to their place on the first two
 * principal components; the colour switches between the K-Means cluster and the true type.
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
    const hex = (h: string) => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)]
    const cA = K.points.map((p) => hex(CLUSTER_COLOR[p[2] % CLUSTER_COLOR.length]))
    const cB = K.points.map((p) => hex(TON_COLOR[ORDER[p[3]]]))

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
      const pad = 24
      K.points.forEach((p, i) => {
        const x = pad + (rand[i][0] + (p[0] - rand[i][0]) * t) * (w - 2 * pad)
        const y = pad + (1 - (rand[i][1] + (p[1] - rand[i][1]) * t)) * (h - 2 * pad)
        const a = cA[i], b = cB[i]
        ctx.fillStyle = `rgba(${a[0] + (b[0] - a[0]) * mix},${a[1] + (b[1] - a[1]) * mix},${a[2] + (b[2] - a[2]) * mix},${0.35 + 0.4 * t})`
        ctx.beginPath()
        ctx.arc(x, y, 2, 0, Math.PI * 2)
        ctx.fill()
      })
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
    <div ref={wrap} className="relative aspect-[4/3] w-full overflow-hidden rounded-xl border border-hairline-strong bg-paper">
      <div className="grid-bg absolute inset-0" />
      <canvas ref={canvas} className="absolute inset-0 h-full w-full" role="img"
        aria-label={`${K.points.length} flows plotted on the first two principal components, coloured by ${colorBy}`} />
      <span className="num absolute bottom-3 left-4 text-[0.65rem] text-ink-dim">PC1 · {pct(K.pcaVariance[0])} of variance</span>
      <span className="num absolute right-4 top-3 text-[0.65rem] text-ink-dim">PC2 · {pct(K.pcaVariance[1])}</span>
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
      <line x1={x(K.k)} x2={x(K.k)} y1={8} y2={h - 22} stroke="#e5481c" strokeDasharray="3 3" />
      <path d={e.map((d, i) => `${i ? 'L' : 'M'}${x(d.k)} ${ys(d.silhouette)}`).join('')} fill="none" stroke="#c98a00" strokeDasharray="3 3" />
      <path d={e.map((d, i) => `${i ? 'L' : 'M'}${x(d.k)} ${y(d.inertia)}`).join('')} fill="none" stroke="#16140f" strokeWidth="1.8" />
      {e.map((d) => (
        <g key={d.k}>
          <circle cx={x(d.k)} cy={y(d.inertia)} r={d.k === K.k ? 5 : 2.5} fill={d.k === K.k ? '#e5481c' : '#efeae0'} stroke="#16140f" />
          <text x={x(d.k)} y={h - 6} fontSize="9" textAnchor="middle" fill={d.k === K.k ? '#16140f' : 'rgba(22,20,15,0.45)'} fontFamily="JetBrains Mono">{d.k}</text>
        </g>
      ))}
    </svg>
  )
}

export function Clusters() {
  const [colorBy, setColorBy] = useState<ColorBy>('cluster')
  const [km, ag] = K.metrics
  const stab = K.stability

  return (
    <Chapter
      id="clusters"
      index="07"
      eyebrow={`Clustering · ${T.overview.name} · no labels`}
      title={<>Let the traffic<br />sort itself</>}
      deck={<>Hide the labels and ask K-Means and agglomerative clustering to group {int(K.sample)} flows by behaviour alone. The elbow of the curve says {K.k} groups. Then recolour the map by the true type, which the algorithms never saw.</>}
    >
      <div className="grid gap-4 lg:grid-cols-[minmax(0,7fr)_minmax(0,5fr)]">
        <div className="grain grid content-start gap-4 rounded-2xl border border-hairline-strong bg-panel p-5 md:p-6">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <span className="eyebrow">Flows on two principal components</span>
            <div className="flex gap-1 rounded-full border border-hairline-strong bg-paper p-1" role="tablist">
              {([['cluster', 'K-Means cluster'], ['class', 'True type']] as const).map(([k, l]) => (
                <button key={k} type="button" role="tab" aria-selected={colorBy === k} onClick={() => setColorBy(k)}
                  className={cn('rounded-full px-3 py-1.5 text-xs font-medium transition', colorBy === k ? 'bg-ink text-paper' : 'text-ink-muted hover:text-ink')}>{l}</button>
              ))}
            </div>
          </div>
          <Field colorBy={colorBy} />
          <div className="flex flex-wrap gap-x-4 gap-y-1.5 text-xs text-ink-muted">
            {colorBy === 'cluster'
              ? K.clusters.map((c) => <span key={c.id} className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-full" style={{ background: CLUSTER_COLOR[c.id] }} />{c.id + 1}: {NAME(c)}</span>)
              : ORDER.map((n) => <span key={n} className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-full" style={{ background: TON_COLOR[n] }} />{tonLabel(n)}</span>)}
          </div>
        </div>

        <div className="grid content-start gap-4">
          <div className="grain grid gap-3 rounded-2xl border border-hairline-strong bg-panel p-6">
            <div className="flex items-baseline justify-between gap-4">
              <span className="eyebrow">Choosing k</span>
              <span className="text-xs text-ink-dim"><span className="text-ink">inertia</span> · <span className="text-amber">silhouette</span></span>
            </div>
            <Elbow />
            <div className="grid grid-cols-3 gap-px overflow-hidden rounded-lg bg-hairline-strong text-center text-xs">
              {[['', 'K-Means', 'Agglomerative'], ['Silhouette ↑', km.silhouette.toFixed(3), ag.silhouette.toFixed(3)],
                ['Davies-Bouldin ↓', km.daviesBouldin.toFixed(2), ag.daviesBouldin.toFixed(2)], ['Calinski-Harabasz ↑', int(km.calinskiHarabasz), int(ag.calinskiHarabasz)]].map((row, i) =>
                row.map((cell, j) => (
                  <span key={`${i}-${j}`} className={cn('bg-paper px-2 py-2', i === 0 || j === 0 ? 'text-ink-dim' : 'num text-ink', j === 0 && 'text-left')}>{cell}</span>
                )))}
            </div>
          </div>
          <div className="grain grid gap-3 rounded-2xl border border-hairline-strong bg-panel p-6">
            <span className="eyebrow">Linkage strategies, agglomerative</span>
            {K.linkages.map((l) => (
              <div key={l.linkage} className="grid grid-cols-[5rem_minmax(0,1fr)_auto] items-center gap-3 text-xs">
                <span className={l.largest < 0.9 ? 'font-semibold' : 'text-ink-muted'}>{l.linkage}</span>
                <span className="relative h-2.5 rounded-sm bg-panel-2" title="share of flows in the largest cluster">
                  <span className={cn('absolute inset-y-0 left-0 rounded-sm', l.largest < 0.9 ? 'bg-signal' : 'bg-red')} style={{ width: `${l.largest * 100}%` }} />
                </span>
                <span className="num">{pct(l.largest)} in one cluster</span>
              </div>
            ))}
            <p className="text-xs leading-relaxed text-ink-dim">Complete, average and single linkage score higher on silhouette only because they put almost every flow in one cluster. Ward gives balanced groups.</p>
          </div>
          {stab.ari && (
            <div className="screen grid grid-cols-2 gap-4 rounded-2xl p-6">
              <div className="grid gap-1"><span className="num text-2xl text-phosphor">{stab.silhouette[0].toFixed(3)} <span className="text-sm text-ink-dim">± {stab.silhouette[1].toFixed(3)}</span></span><span className="eyebrow">silhouette, 5 resamples</span></div>
              <div className="grid gap-1"><span className="num text-2xl text-phosphor">{stab.ari[0].toFixed(3)}</span><span className="eyebrow">agreement between runs (ARI)</span></div>
            </div>
          )}
        </div>
      </div>

      <div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {K.clusters.filter((c) => c.size >= 50).map((c) => (
          <div key={c.id} className="grain grid content-start gap-3 rounded-xl border border-hairline-strong bg-panel p-5">
            <div className="flex items-baseline justify-between gap-3">
              <span className="flex items-center gap-2 font-semibold"><span className="h-3 w-3 rounded-full" style={{ background: CLUSTER_COLOR[c.id] }} />Cluster {c.id + 1}</span>
              <span className="num text-xs text-ink-dim">{int(c.size)} flows</span>
            </div>
            <div className="flex h-3 overflow-hidden rounded-sm bg-panel-2">
              {c.mix.map((m) => <span key={m.name} title={`${tonLabel(m.name)} ${pct(m.share)}`} style={{ width: `${m.share * 100}%`, background: TON_COLOR[m.name] }} />)}
            </div>
            <ul className="grid gap-0.5 text-xs">
              {c.mix.slice(0, 4).map((m) => (
                <li key={m.name} className="flex justify-between gap-3"><span style={{ color: TON_COLOR[m.name] }}>{tonLabel(m.name)}</span><span className="num text-ink-muted">{pct(m.share)}</span></li>
              ))}
            </ul>
          </div>
        ))}
        <div className="grid content-center gap-2 rounded-xl border border-dashed border-hairline-strong p-5 text-sm text-ink-muted">
          <span className="eyebrow">Cluster sizes</span>
          {K.clusters.map((c) => <span key={c.id} className="num text-xs">Cluster {c.id + 1}: {int(c.size)}{c.size < 50 ? ' (a handful of outlier flows)' : ''}</span>)}
        </div>
      </div>
    </Chapter>
  )
}
