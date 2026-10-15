import { useMemo, useState } from 'react'
import { Chapter } from '@/components/chapter'
import { T, int, tonLabel, type CmpCluster, type CmpModel } from '@/lib/data'
import { cn } from '@/lib/utils'

const C = T.comparison.classification
const K = T.comparison.clustering

const FAMILY_COLOR: Record<string, string> = {
  'Bagging ensemble': '#0f6e5a', 'Boosting ensemble': '#e5481c', 'Single tree': '#c98a00', 'Distance-based': '#2f5d9e',
  'Neural network': '#7b3fa0', Kernel: '#c2185b', Linear: '#5b4636', Probabilistic: '#4a4a4a',
}

/* ================================================================== classification: scoreboard */

type Col = { key: keyof CmpModel; label: string; low?: boolean; fmt: (v: number) => string }
const COLS: Col[] = [
  { key: 'acc', label: 'Accuracy', fmt: (v) => v.toFixed(3) },
  { key: 'precision', label: 'Precision', fmt: (v) => v.toFixed(3) },
  { key: 'recall', label: 'Recall', fmt: (v) => v.toFixed(3) },
  { key: 'f1w', label: 'F1 weighted', fmt: (v) => v.toFixed(3) },
  { key: 'f1m', label: 'F1 macro', fmt: (v) => v.toFixed(3) },
  { key: 'auc', label: 'ROC-AUC', fmt: (v) => v.toFixed(3) },
  { key: 'trainTime', label: 'Train s', low: true, fmt: (v) => (v < 10 ? v.toFixed(1) : int(v)) },
]

/** rank of every model on every column, 1 = best */
const RANKS = Object.fromEntries(COLS.map((c) => {
  const sorted = [...C.rows].sort((a, b) => (c.low ? 1 : -1) * ((a[c.key] as number) - (b[c.key] as number)))
  return [c.key, Object.fromEntries(C.rows.map((r) => [r.key, sorted.findIndex((s) => (s[c.key] as number) === (r[c.key] as number)) + 1]))]
})) as Record<string, Record<string, number>>
const MEAN_RANK = Object.fromEntries(C.rows.map((r) => [r.key, COLS.slice(0, 6).reduce((s, c) => s + RANKS[c.key][r.key], 0) / 6]))
/** green for the top ranks, red for the bottom ones, pale in the middle */
const rankColor = (rank: number) => {
  const t = (rank - 1) / (C.rows.length - 1)
  const a = 0.06 + 0.5 * Math.abs(t - 0.5) * 2
  return t < 0.5 ? `rgba(15,110,90,${a})` : `rgba(229,72,28,${a})`
}

function Scoreboard() {
  const [sortBy, setSortBy] = useState<string>('f1w')
  const rows = useMemo(() => {
    if (sortBy === 'mean') return [...C.rows].sort((a, b) => MEAN_RANK[a.key] - MEAN_RANK[b.key])
    const col = COLS.find((c) => c.key === sortBy)!
    return [...C.rows].sort((a, b) => (col.low ? 1 : -1) * ((a[col.key] as number) - (b[col.key] as number)))
  }, [sortBy])
  return (
    <div className="grain grid content-start gap-4 rounded-2xl border border-hairline-strong bg-panel p-5 md:p-7 lg:col-span-2">
      <div className="flex flex-wrap items-baseline justify-between gap-3">
        <span className="eyebrow">Every configuration on every metric · colour = rank</span>
        <span className="text-xs text-ink-dim">click a column to sort · green is rank 1, red is last</span>
      </div>
      <div className="overflow-x-auto">
        <table className="num w-full min-w-[760px] border-separate border-spacing-[3px] text-[0.72rem]">
          <thead>
            <tr>
              <th className="text-left font-normal text-ink-dim">model</th>
              {[...COLS.map((c) => [c.key as string, c.label]), ['mean', 'Mean rank']].map(([k, l]) => (
                <th key={k} className="font-normal">
                  <button type="button" onClick={() => setSortBy(k)} aria-pressed={sortBy === k}
                    className={cn('w-full rounded px-1 py-1 transition', sortBy === k ? 'bg-ink text-paper' : 'text-ink-dim hover:text-ink')}>{l}</button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.key}>
                <td className="whitespace-nowrap pr-2 text-left font-sans text-[0.8rem]">
                  <span className="mr-2 inline-block h-2 w-2 rounded-full align-middle" style={{ background: FAMILY_COLOR[r.family] }} />
                  {r.model} <span className="text-ink-dim">· {r.part}</span>
                </td>
                {COLS.map((c) => (
                  <td key={c.key} className="rounded-[3px] px-1.5 py-1.5 text-center" title={`rank ${RANKS[c.key][r.key]} of ${C.rows.length}`}
                    style={{ background: rankColor(RANKS[c.key][r.key]) }}>{c.fmt(r[c.key] as number)}</td>
                ))}
                <td className="rounded-[3px] px-1.5 py-1.5 text-center font-semibold" style={{ background: rankColor(Math.round(MEAN_RANK[r.key])) }}>
                  {MEAN_RANK[r.key].toFixed(1)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-[0.7rem] text-ink-muted">
        {Object.entries(FAMILY_COLOR).filter(([f]) => C.rows.some((r) => r.family === f)).map(([f, col]) => (
          <span key={f}><span className="mr-1.5 inline-block h-2 w-2 rounded-full" style={{ background: col }} />{f}</span>
        ))}
      </div>
    </div>
  )
}

/* ================================================================== classification: per-class F1 */

function PerClassGrid() {
  const [hover, setHover] = useState<[number, number] | null>(null)
  const rows = C.rows
  const h = hover ?? [0, 0]
  return (
    <div className="grain grid content-start gap-4 rounded-2xl border border-hairline-strong bg-panel p-5 md:p-7 lg:col-span-2">
      <div className="flex flex-wrap items-baseline justify-between gap-3">
        <span className="eyebrow">Which algorithm fails on which attack · F1 per traffic type</span>
        <span className="text-xs text-ink-dim">rows ranked by weighted F1 · test support under each type</span>
      </div>
      <div className="overflow-x-auto">
        <div className="grid min-w-[760px] gap-[3px]" style={{ gridTemplateColumns: `11rem repeat(${C.classes.length}, minmax(0, 1fr))` }} onMouseLeave={() => setHover(null)}>
          <span />
          {C.classes.map((c, j) => (
            <span key={c} className={cn('grid pb-1 text-center text-[0.62rem] leading-tight', j === h[1] && hover ? 'text-ink' : 'text-ink-dim')}>
              <span className="uppercase tracking-wider">{tonLabel(c)}</span><span className="num">n={int(C.support[c])}</span>
            </span>
          ))}
          {rows.map((r, i) => (
            <div key={r.key} className="contents">
              <span className={cn('truncate pr-2 text-right text-xs leading-8', hover && i === h[0] ? 'font-medium text-ink' : 'text-ink-muted')}>{r.model}</span>
              {r.perClass.map((v, j) => (
                <button key={j} type="button" onMouseEnter={() => setHover([i, j])} onFocus={() => setHover([i, j])}
                  aria-label={`${r.model}, ${tonLabel(C.classes[j])}: F1 ${v.toFixed(2)}`}
                  className={cn('num grid min-h-8 place-items-center rounded-[3px] text-[0.62rem]', hover && i === h[0] && j === h[1] && 'outline outline-2 outline-offset-1 outline-ink')}
                  style={{ background: `rgba(${v >= 0.5 ? '15,110,90' : '229,72,28'},${0.1 + 0.8 * Math.abs(v - 0.5) * 2})`, color: Math.abs(v - 0.5) > 0.3 ? '#efeae0' : 'rgba(22,20,15,0.8)' }}>
                  {v.toFixed(2)}
                </button>
              ))}
            </div>
          ))}
        </div>
      </div>
      <p className="min-h-[1.5rem] text-sm text-ink-muted" aria-live="polite">
        {hover
          ? <><b className="text-ink">{rows[h[0]].model}</b> scores F1 <span className="num">{rows[h[0]].perClass[h[1]].toFixed(3)}</span> on {tonLabel(C.classes[h[1]])}; the best on that type is {(() => { const b = [...rows].sort((a, b) => b.perClass[h[1]] - a.perClass[h[1]])[0]; return `${b.model} (${b.perClass[h[1]].toFixed(3)})` })()}.</>
          : 'Hover a cell to compare it with the best model for that traffic type.'}
      </p>
    </div>
  )
}

/* ================================================================== classification: cost vs benefit with intervals */

function CostBenefit() {
  const [hover, setHover] = useState<string | null>(null)
  const w = 560, h = 300, pad = { l: 46, r: 30, t: 14, b: 34 }
  const tmin = Math.min(...C.rows.map((r) => r.trainTime)), tmax = Math.max(...C.rows.map((r) => r.trainTime))
  const lt = (v: number) => Math.log10(Math.max(v, 0.01))
  const x = (v: number) => pad.l + ((lt(v) - lt(tmin) + 0.2) / (lt(tmax) - lt(tmin) + 0.4)) * (w - pad.l - pad.r)
  const fmin = Math.min(...C.rows.map((r) => r.ci[0])) - 0.02, fmax = Math.max(...C.rows.map((r) => r.ci[1])) + 0.01
  const y = (v: number) => pad.t + (1 - (v - fmin) / (fmax - fmin)) * (h - pad.t - pad.b)
  const front = [...C.rows].sort((a, b) => a.trainTime - b.trainTime).filter((r, i, arr) => arr.slice(0, i).every((p) => p.f1w < r.f1w))
  const ticks = [0.1, 1, 10, 100, 1000].filter((t) => t >= tmin / 3 && t <= tmax * 3)
  const sel = C.rows.find((r) => r.key === hover)
  return (
    <div className="grain grid content-start gap-4 rounded-2xl border border-hairline-strong bg-panel p-5 md:p-7">
      <span className="eyebrow">Cost against benefit · weighted F1 with 95% bootstrap interval</span>
      <svg viewBox={`0 0 ${w} ${h}`} className="w-full" role="img" aria-label="Weighted F1 against training time for every configuration">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={x(t)} x2={x(t)} y1={pad.t} y2={h - pad.b} stroke="rgba(22,20,15,0.08)" />
            <text x={x(t)} y={h - pad.b + 14} fontSize="9" textAnchor="middle" fill="rgba(22,20,15,0.5)" fontFamily="JetBrains Mono">{t}s</text>
          </g>
        ))}
        {[0.3, 0.5, 0.7, 0.8, 0.9, 1].filter((v) => v > fmin && v < fmax).map((v) => (
          <g key={v}>
            <line x1={pad.l} x2={w - pad.r} y1={y(v)} y2={y(v)} stroke="rgba(22,20,15,0.08)" />
            <text x={pad.l - 6} y={y(v) + 3} fontSize="9" textAnchor="end" fill="rgba(22,20,15,0.5)" fontFamily="JetBrains Mono">{v.toFixed(1)}</text>
          </g>
        ))}
        <path d={front.map((r, i) => `${i ? 'L' : 'M'}${x(r.trainTime)} ${y(r.f1w)}`).join('')} fill="none" stroke="#16140f" strokeDasharray="4 3" strokeWidth="1" />
        {C.rows.map((r) => (
          <g key={r.key} onMouseEnter={() => setHover(r.key)} onMouseLeave={() => setHover(null)} className="cursor-pointer">
            <line x1={x(r.trainTime)} x2={x(r.trainTime)} y1={y(r.ci[0])} y2={y(r.ci[1])} stroke={FAMILY_COLOR[r.family]} strokeWidth="2" opacity="0.6" />
            <circle cx={x(r.trainTime)} cy={y(r.f1w)} r={hover === r.key ? 7 : 5} fill={FAMILY_COLOR[r.family]} stroke="#efeae0" strokeWidth="1.5" />
          </g>
        ))}
        <text x={(w + pad.l) / 2} y={h - 4} fontSize="9" textAnchor="middle" fill="rgba(22,20,15,0.55)" fontFamily="JetBrains Mono">training time (log scale)</text>
      </svg>
      <p className="min-h-[2.8rem] text-sm leading-relaxed text-ink-muted" aria-live="polite">
        {sel
          ? <><b className="text-ink">{sel.model}</b>: weighted F1 <span className="num">{sel.f1w.toFixed(3)}</span> (95% interval {sel.ci[0].toFixed(3)}–{sel.ci[1].toFixed(3)}), trained in <span className="num">{sel.trainTime < 10 ? sel.trainTime.toFixed(1) : int(sel.trainTime)}s</span>.</>
          : <>Dashed line: models no other model beats on both speed and score ({front.map((r) => r.model).join(', ')}). Hover a point for its numbers.</>}
      </p>
    </div>
  )
}

function GapReal() {
  const best = C.rows[0]
  const others = C.rows.slice(1)
  return (
    <div className="screen grain grid content-start gap-4 rounded-2xl p-5 md:p-7">
      <span className="eyebrow text-red">Is the gap real? · McNemar's test against {best.model}</span>
      <div className="grid gap-1.5">
        {others.map((r) => {
          const p = r.mcnemarP ?? 1
          const sig = p < 0.05
          return (
            <div key={r.key} className="grid grid-cols-[minmax(0,1fr)_4.2rem_5.6rem] items-center gap-3 text-xs">
              <span className="truncate text-ink-muted">{r.model}</span>
              <span className="num text-right">{(best.f1w - r.f1w >= 0 ? '−' : '+') + Math.abs(best.f1w - r.f1w).toFixed(3)}</span>
              <span className={cn('num rounded px-1.5 py-0.5 text-center text-[0.65rem]', sig ? 'bg-red text-paper' : 'bg-panel-2 text-ink-dim')}>
                {p < 0.0001 ? 'p < 0.0001' : `p = ${p.toFixed(p < 0.01 ? 4 : 3)}`}
              </span>
            </div>
          )
        })}
      </div>
      <p className="text-xs leading-relaxed text-ink-dim">
        Middle column: weighted F1 behind {best.model}. McNemar's test looks only at the test flows where exactly one of the two models is right; red marks p below 0.05, where the difference in errors is unlikely to be chance.
      </p>
    </div>
  )
}

/* ================================================================== clustering */

type KMetric = 'silhouette' | 'daviesBouldin' | 'calinskiHarabasz' | 'ari'
const KMETRICS: { key: KMetric; label: string; low?: boolean }[] = [
  { key: 'silhouette', label: 'Silhouette ↑' },
  { key: 'daviesBouldin', label: 'Davies-Bouldin ↓', low: true },
  { key: 'calinskiHarabasz', label: 'Calinski-Harabasz ↑' },
  { key: 'ari', label: 'ARI vs true type ↑' },
]
const SWEEP_COLOR: Record<string, string> = { 'K-Means': '#e5481c', 'Agglomerative (ward)': '#2f5d9e', 'Gaussian Mixture': '#0f6e5a', BIRCH: '#c98a00' }

function ClusterTable() {
  const cols: { key: keyof CmpCluster; label: string; low?: boolean; d: number }[] = [
    { key: 'silhouette', label: 'Silhouette ↑', d: 3 }, { key: 'daviesBouldin', label: 'Davies-Bouldin ↓', low: true, d: 3 },
    { key: 'calinskiHarabasz', label: 'Calinski-H. ↑', d: 0 }, { key: 'ari', label: 'ARI ↑', d: 3 }, { key: 'nmi', label: 'NMI ↑', d: 3 },
    { key: 'purity', label: 'Purity ↑', d: 3 }, { key: 'largest', label: 'Largest cluster', d: 2 }, { key: 'time', label: 'Fit s', d: 2 },
  ]
  const best = (c: (typeof cols)[number]) => {
    if (c.key === 'largest' || c.key === 'time') return null
    const v = K.rows.map((r) => r[c.key] as number | null).filter((x): x is number => x != null)
    return c.low ? Math.min(...v) : Math.max(...v)
  }
  return (
    <div className="grain grid content-start gap-4 rounded-2xl border border-hairline-strong bg-panel p-5 md:p-7 lg:col-span-2">
      <div className="flex flex-wrap items-baseline justify-between gap-3">
        <span className="eyebrow">Every clustering on the same 6,000 flows</span>
        <span className="text-xs text-ink-dim">outlined: best on that metric · ARI, NMI and purity use the true type after fitting</span>
      </div>
      <div className="overflow-x-auto">
        <table className="num w-full min-w-[760px] text-[0.72rem]">
          <thead><tr className="text-ink-dim"><th className="py-1 text-left font-normal">algorithm</th><th className="font-normal">k</th>{cols.map((c) => <th key={c.key} className="font-normal">{c.label}</th>)}</tr></thead>
          <tbody>
            {K.rows.map((r) => (
              <tr key={r.name} className="border-t border-hairline-strong">
                <td className="whitespace-nowrap py-1.5 pr-2 text-left font-sans text-[0.8rem]">
                  {r.name} {!r.required && <span className="chip ml-1">extra</span>}
                </td>
                <td className="text-center">{r.clusters}</td>
                {cols.map((c) => {
                  const v = r[c.key] as number | null
                  const b = best(c)
                  return (
                    <td key={c.key} className="px-1 text-center">
                      <span className={cn('inline-block rounded px-1.5 py-0.5', v != null && v === b && 'outline outline-2 outline-signal', c.key === 'largest' && v != null && v >= 0.9 && 'bg-red text-paper')}>
                        {v == null ? '–' : v.toFixed(c.d)}
                      </span>
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-xs leading-relaxed text-ink-dim">
        Red: one cluster holds 90% or more of the flows. A linkage that puts almost everything in one cluster can still post a high silhouette, which is why cluster balance is shown next to the scores. DBSCAN picks its own number of clusters and leaves {(() => { const d = K.rows.find((r) => r.name === 'DBSCAN'); return d ? `${Math.round(d.noise * 100)}%` : 'some' })()} of the flows as noise.
      </p>
    </div>
  )
}

function Sweep() {
  const [m, setM] = useState<KMetric>('silhouette')
  const w = 520, h = 230, pad = { l: 44, r: 12, t: 12, b: 26 }
  const names = Object.keys(SWEEP_COLOR).filter((n) => K.sweep.some((s) => s.name === n))
  const ks = [...new Set(K.sweep.map((s) => s.k))].sort((a, b) => a - b)
  const vals = K.sweep.map((s) => s[m])
  const lo = Math.min(...vals), hi = Math.max(...vals)
  const x = (k: number) => pad.l + ((k - ks[0]) / (ks[ks.length - 1] - ks[0])) * (w - pad.l - pad.r)
  const y = (v: number) => pad.t + (1 - (v - lo) / (hi - lo || 1)) * (h - pad.t - pad.b)
  const kBest = T.clustering.k
  return (
    <div className="grain grid content-start gap-4 rounded-2xl border border-hairline-strong bg-panel p-5 md:p-7">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <span className="eyebrow">Do the metrics agree on k?</span>
        <div className="flex flex-wrap gap-1 rounded-full border border-hairline-strong bg-paper p-1" role="tablist">
          {KMETRICS.map((k) => (
            <button key={k.key} type="button" role="tab" aria-selected={m === k.key} onClick={() => setM(k.key)}
              className={cn('rounded-full px-2.5 py-1 text-[0.68rem] font-medium transition', m === k.key ? 'bg-ink text-paper' : 'text-ink-muted hover:text-ink')}>{k.label}</button>
          ))}
        </div>
      </div>
      <svg viewBox={`0 0 ${w} ${h}`} className="w-full" role="img" aria-label={`${KMETRICS.find((k) => k.key === m)!.label} for k from ${ks[0]} to ${ks[ks.length - 1]}`}>
        <line x1={x(kBest)} x2={x(kBest)} y1={pad.t} y2={h - pad.b} stroke="#e5481c" strokeDasharray="3 3" />
        {[lo, (lo + hi) / 2, hi].map((v) => (
          <text key={v} x={pad.l - 6} y={y(v) + 3} fontSize="9" textAnchor="end" fill="rgba(22,20,15,0.5)" fontFamily="JetBrains Mono">{m === 'calinskiHarabasz' ? Math.round(v) : v.toFixed(2)}</text>
        ))}
        {ks.map((k) => <text key={k} x={x(k)} y={h - 8} fontSize="9" textAnchor="middle" fill={k === kBest ? '#16140f' : 'rgba(22,20,15,0.45)'} fontFamily="JetBrains Mono">{k}</text>)}
        {names.map((n) => {
          const pts = K.sweep.filter((s) => s.name === n).sort((a, b) => a.k - b.k)
          return (
            <g key={n}>
              <path d={pts.map((p, i) => `${i ? 'L' : 'M'}${x(p.k)} ${y(p[m])}`).join('')} fill="none" stroke={SWEEP_COLOR[n]} strokeWidth={n === 'K-Means' || n.startsWith('Agglo') ? 2.2 : 1.2} />
              {pts.map((p) => <circle key={p.k} cx={x(p.k)} cy={y(p[m])} r="2.5" fill={SWEEP_COLOR[n]} />)}
            </g>
          )
        })}
      </svg>
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-[0.7rem] text-ink-muted">
        {names.map((n) => <span key={n}><span className="mr-1.5 inline-block h-0.5 w-4 align-middle" style={{ background: SWEEP_COLOR[n] }} />{n}</span>)}
        <span><span className="mr-1.5 inline-block h-0 w-4 border-t border-dashed border-red align-middle" />k = {kBest} from the elbow</span>
      </div>
    </div>
  )
}

function StabilityCost() {
  const maxT = Math.max(...K.runtime.map((r) => r.time))
  const big = Math.max(...K.runtime.map((r) => r.flows))
  const at = K.runtime.filter((r) => r.flows === big).sort((a, b) => b.time - a.time)
  return (
    <div className="grain grid content-start gap-5 rounded-2xl border border-hairline-strong bg-panel p-5 md:p-7">
      <span className="eyebrow">Repeatable and affordable?</span>
      <div className="grid gap-2">
        <span className="text-xs text-ink-muted">Agreement between 5 runs on the same 2,000 flows (ARI, 1 = identical)</span>
        {K.stability.map((s) => (
          <div key={s.name} className="grid grid-cols-[9rem_minmax(0,1fr)_7rem] items-center gap-3 text-xs">
            <span className="truncate text-ink-muted">{s.name}</span>
            <span className="relative h-2.5 rounded-sm bg-panel-2"><span className="absolute inset-y-0 left-0 rounded-sm bg-signal" style={{ width: `${Math.max(0, s.ari) * 100}%` }} /></span>
            <span className="num text-right">{s.ari.toFixed(3)} <span className="text-ink-dim">± {s.std.toFixed(2)}</span></span>
          </div>
        ))}
      </div>
      <div className="grid gap-2">
        <span className="text-xs text-ink-muted">Fit time on {int(big)} flows</span>
        {at.map((r) => (
          <div key={r.name} className="grid grid-cols-[9rem_minmax(0,1fr)_4rem] items-center gap-3 text-xs">
            <span className="truncate text-ink-muted">{r.name}</span>
            <span className="relative h-2.5 rounded-sm bg-panel-2"><span className="absolute inset-y-0 left-0 rounded-sm bg-red" style={{ width: `${Math.max(1, (Math.log10(r.time * 1000 + 1) / Math.log10(maxT * 1000 + 1)) * 100)}%` }} /></span>
            <span className="num text-right">{r.time < 1 ? r.time.toFixed(2) : r.time.toFixed(1)}s</span>
          </div>
        ))}
        <span className="text-[0.65rem] text-ink-dim">bars on a log scale · agglomerative clustering stores every pairwise distance, so its cost grows with the square of the flows</span>
      </div>
    </div>
  )
}

export function Compare() {
  const best = C.rows[0]
  const km = K.rows.find((r) => r.name === 'K-Means')
  return (
    <Chapter
      id="compare"
      index="08"
      eyebrow="Comparative study · both tracks"
      title={<>Side by side,<br />metric by metric</>}
      deck={<>All {C.rows.length} classification configurations and {K.rows.length} clusterings, compared on more than one number: every metric, every traffic type, training cost, statistical significance, agreement on k and repeatability. {best.model} leads on weighted F1 ({best.f1w.toFixed(3)}); K-Means scores silhouette {km?.silhouette?.toFixed(3)} on the shared clustering sample.</>}
    >
      <div className="grid gap-4 lg:grid-cols-2">
        <Scoreboard />
        <PerClassGrid />
        <CostBenefit />
        <GapReal />
      </div>
      <div className="mt-10 grid gap-4 lg:grid-cols-2">
        <ClusterTable />
        <Sweep />
        <StabilityCost />
      </div>
    </Chapter>
  )
}
