import { useEffect, useMemo, useRef, useState } from 'react'
import { Chapter } from '@/components/chapter'
import { Slider } from '@/components/ui/slider'
import { T, TON_COLOR, int, pct } from '@/lib/data'
import { cn } from '@/lib/utils'

const C = T.classification
const ORDER = C.allClasses

/* ================================================================== Game 1: tune it yourself */

type GridRow = { n_estimators: number; learning_rate: number; max_depth: number; cv: number; cvStd: number; test: number; testAcc: number }
const GRID: GridRow[] = C.tuning?.grid ?? []
const TREES = [50, 100, 200]
const RATES = [0.05, 0.1, 0.2]
const DEPTHS = [3, 5]
const BUDGET = 6
const team = C.models.find((m) => m.key === C.best)!

function Choice({ label, options, value, onChange }: { label: string; options: number[]; value: number; onChange: (v: number) => void }) {
  return (
    <div className="grid gap-1.5">
      <span className="eyebrow">{label}</span>
      <div className="flex gap-1 rounded-full border border-hairline-strong bg-paper p-1">
        {options.map((o) => (
          <button key={o} type="button" onClick={() => onChange(o)}
            className={cn('num flex-1 rounded-full px-3 py-1.5 text-sm transition', value === o ? 'bg-ink text-paper' : 'text-ink-muted hover:text-ink')}>
            {o}
          </button>
        ))}
      </div>
    </div>
  )
}

function TuneGame() {
  const [n, setN] = useState(100)
  const [lr, setLr] = useState(0.1)
  const [depth, setDepth] = useState(3)
  const [tries, setTries] = useState<GridRow[]>([])
  const [training, setTraining] = useState(false)
  const [revealed, setRevealed] = useState(false)
  const timer = useRef<number | undefined>(undefined)
  useEffect(() => () => window.clearTimeout(timer.current), [])

  if (!GRID.length) return null
  const row = (a: number, b: number, c: number) => GRID.find((g) => g.n_estimators === a && g.learning_rate === b && g.max_depth === c)
  const best = tries.reduce<GridRow | null>((b, t) => (!b || t.cv > b.cv ? t : b), null)
  const gridBestCv = GRID.reduce((b, g) => (g.cv > b.cv ? g : b), GRID[0])
  const gridBestTest = GRID.reduce((b, g) => (g.test > b.test ? g : b), GRID[0])
  const cvMin = Math.min(...GRID.map((g) => g.cv)), cvMax = Math.max(...GRID.map((g) => g.cv))
  const testMin = Math.min(...GRID.map((g) => g.test))

  const train = () => {
    const r = row(n, lr, depth)
    if (!r || training || tries.length >= BUDGET) return
    setTraining(true)
    timer.current = window.setTimeout(() => { setTries((t) => [...t, r]); setTraining(false) }, 1100)
  }
  const reset = () => { setTries([]); setRevealed(false) }

  return (
    <div className="grain grid content-start gap-5 rounded-2xl border border-hairline-strong bg-panel p-5 md:p-7">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="grid gap-1">
          <span className="eyebrow">Game 1 · Tune it yourself</span>
          <h3 className="display text-[1.4rem] leading-tight">Tune Gradient Boosting</h3>
        </div>
        <div className="grid justify-items-end">
          <span className="num text-2xl leading-none">{best ? best.cv.toFixed(4) : '–'}</span>
          <span className="text-[0.65rem] uppercase tracking-wider text-ink-dim">your best CV F1 · {BUDGET - tries.length} tries left</span>
        </div>
      </div>
      <p className="text-sm leading-relaxed text-ink-muted">
        Pick the settings, train, and see the real 3-fold cross-validated weighted F1 they get. You have {BUDGET} tries to find the best one, then lock it in to see how it does on machines it has never seen.
      </p>

      <div className="grid gap-3 sm:grid-cols-3">
        <Choice label="Trees" options={TREES} value={n} onChange={setN} />
        <Choice label="Learning rate" options={RATES} value={lr} onChange={setLr} />
        <Choice label="Tree depth" options={DEPTHS} value={depth} onChange={setDepth} />
      </div>
      <div className="flex flex-wrap gap-2">
        <button type="button" onClick={train} disabled={training || tries.length >= BUDGET || revealed}
          className="rounded-full bg-ink px-5 py-2.5 text-sm font-semibold text-paper transition hover:bg-red disabled:opacity-50">
          {training ? 'Training 3 folds…' : 'Train and cross-validate'}
        </button>
        {best && !revealed && <button type="button" onClick={() => setRevealed(true)} className="rounded-full border border-ink px-5 py-2.5 text-sm font-semibold">Lock in my best</button>}
        {(revealed || tries.length > 0) && <button type="button" onClick={reset} className="rounded-full px-3 py-2.5 text-sm text-ink-muted underline underline-offset-4">start over</button>}
      </div>

      {training && (
        <div className="grid grid-cols-3 gap-2" aria-hidden>
          {[0, 1, 2].map((f) => (
            <div key={f} className="h-2 overflow-hidden rounded-full bg-panel-2">
              <div className="h-full origin-left animate-pulse bg-red" style={{ animationDelay: `${f * 0.15}s` }} />
            </div>
          ))}
        </div>
      )}

      {tries.length > 0 && (
        <ol className="grid gap-1.5">
          {tries.map((t, i) => (
            <li key={i} className="grid grid-cols-[1.4rem_minmax(0,1.3fr)_minmax(0,1fr)_3.6rem] items-center gap-3 text-xs">
              <span className="num text-ink-dim">{i + 1}</span>
              <span className="num text-ink-muted">{t.n_estimators} trees · lr {t.learning_rate} · depth {t.max_depth}</span>
              <span className="relative h-2.5 rounded-sm bg-panel-2">
                <span className={cn('absolute inset-y-0 left-0 rounded-sm', t === best ? 'bg-red' : 'bg-ink/60')} style={{ width: `${8 + ((t.cv - cvMin) / (cvMax - cvMin || 1)) * 92}%` }} />
              </span>
              <span className="num text-right">{t.cv.toFixed(4)}</span>
            </li>
          ))}
        </ol>
      )}

      {revealed && best && (
        <div className="grid gap-4 rounded-xl border border-hairline-strong bg-paper p-4">
          <div className="grid grid-cols-2 gap-4">
            <div className="grid gap-0.5"><span className="eyebrow">CV F1, same machines</span><span className="num text-2xl">{best.cv.toFixed(4)}</span></div>
            <div className="grid gap-0.5"><span className="eyebrow text-red">Test F1, unseen machines</span><span className="num text-2xl text-red">{best.test.toFixed(4)}</span></div>
          </div>
          <p className="text-sm leading-relaxed text-ink-muted">
            {best === gridBestCv ? 'You found the top cross-validation setting. ' : `The top cross-validation setting was ${gridBestCv.n_estimators} trees, learning rate ${gridBestCv.learning_rate}, depth ${gridBestCv.max_depth} (${gridBestCv.cv.toFixed(4)}). `}
            The cross-validation folds share machines with each other, the test set does not. On unseen machines the best of all 18 settings was {gridBestTest.n_estimators} trees, learning rate {gridBestTest.learning_rate}, depth {gridBestTest.max_depth} ({gridBestTest.test.toFixed(4)}).
          </p>
          <div className="overflow-x-auto">
            <table className="num w-full min-w-[440px] text-center text-[0.7rem]">
              <thead><tr className="text-ink-dim"><th className="py-1 text-left font-normal">depth · lr</th>{TREES.map((t) => <th key={t} className="font-normal">{t} trees</th>)}</tr></thead>
              <tbody>
                {DEPTHS.flatMap((d) => RATES.map((l) => (
                  <tr key={`${d}-${l}`}>
                    <td className="py-0.5 text-left text-ink-muted">{d} · {l}</td>
                    {TREES.map((t) => {
                      const g = row(t, l, d)!
                      return (
                        <td key={t} className="p-0.5">
                          <span className={cn('block rounded px-1 py-1', g === best && 'outline outline-2 outline-red')}
                            style={{ background: `rgba(15,110,90,${0.08 + 0.55 * ((g.test - testMin) / (gridBestTest.test - testMin || 1))})` }}>
                            {g.cv.toFixed(3)} / <b>{g.test.toFixed(3)}</b>
                          </span>
                        </td>
                      )
                    })}
                  </tr>
                )))}
              </tbody>
            </table>
            <p className="mt-2 text-[0.7rem] text-ink-dim">Each cell: CV F1 / <b>test F1 on unseen machines</b>. Darker green is better on test; your pick is outlined.</p>
          </div>
        </div>
      )}
      <p className="text-xs leading-relaxed text-ink-dim">
        For speed, every setting here was trained on a {int(C.tuning?.sampleRows ?? 0)}-flow sample of the training machines. For comparison, the team's tuned model, trained on all {int(C.trainRows)} training flows, scores {team.f1w.toFixed(4)} on the same unseen machines: more data from the same few machines does not automatically mean better results on new ones.
      </p>
    </div>
  )
}

/* ================================================================== Game 2: firewall defender */

type Packet = { p: number; cls: number; x: number; y: number; decided: boolean; blocked: boolean; vy: number; alpha: number }
const GAME_SECONDS = 45
const SCORE = { tp: 10, tn: 1, fp: -15, fn: -25 }

/** Precision and recall of "block if P(attack) >= t" over the whole stream, for the trade-off chart. */
function useCurve() {
  return useMemo(() => {
    const flows = C.firewall.flows
    return Array.from({ length: 19 }, (_, k) => {
      const t = 0.05 + k * 0.05
      let tp = 0, fp = 0, fn = 0
      for (const [p, c] of flows) {
        const attack = ORDER[c] !== 'normal'
        if (p >= t) { if (attack) tp++; else fp++ } else if (attack) fn++
      }
      return { t, precision: tp / Math.max(1, tp + fp), recall: tp / Math.max(1, tp + fn) }
    })
  }, [])
}

function TradeOff({ t }: { t: number }) {
  const curve = useCurve()
  const w = 260, h = 120
  // y-axis zoomed to the range the curves actually cover, so the trade-off is visible
  const lo = Math.max(0, Math.floor(Math.min(...curve.flatMap((d) => [d.precision, d.recall])) * 20) / 20 - 0.05)
  const x = (v: number) => 10 + ((v - 0.05) / 0.9) * (w - 20)
  const y = (v: number) => h - 14 - ((v - lo) / (1 - lo)) * (h - 26)
  const path = (key: 'precision' | 'recall') => curve.map((d, i) => `${i ? 'L' : 'M'}${x(d.t)} ${y(d[key])}`).join('')
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full" role="img" aria-label="Precision and recall against the blocking threshold">
      <path d={path('recall')} fill="none" stroke="#e5481c" strokeWidth="2" />
      <path d={path('precision')} fill="none" stroke="#5fd39a" strokeWidth="2" />
      <line x1={x(t)} x2={x(t)} y1={6} y2={h - 12} stroke="#efeae0" strokeDasharray="3 3" />
      <text x={10} y={h - 2} fontSize="8" fill="rgba(239,234,224,0.5)" fontFamily="JetBrains Mono">0.05</text>
      <text x={w - 10} y={10} fontSize="8" textAnchor="end" fill="rgba(239,234,224,0.5)" fontFamily="JetBrains Mono">1.00</text>
      <text x={w - 10} y={h - 16} fontSize="8" textAnchor="end" fill="rgba(239,234,224,0.5)" fontFamily="JetBrains Mono">{lo.toFixed(2)}</text>
      <text x={w - 10} y={h - 2} fontSize="8" textAnchor="end" fill="rgba(239,234,224,0.5)" fontFamily="JetBrains Mono">0.95 threshold</text>
    </svg>
  )
}

function Firewall() {
  const canvas = useRef<HTMLCanvasElement>(null)
  const [threshold, setThreshold] = useState(0.5)
  const [running, setRunning] = useState(false)
  const [time, setTime] = useState(GAME_SECONDS)
  const [stats, setStats] = useState({ tp: 0, fp: 0, fn: 0, tn: 0, score: 0 })
  const tRef = useRef(threshold)
  tRef.current = threshold
  const game = useRef({ packets: [] as Packet[], cursor: 0, spawnAcc: 0, elapsed: 0, flash: 0, flashColor: '#e5481c' })

  const reset = () => {
    game.current = { packets: [], cursor: Math.floor(Math.random() * C.firewall.flows.length), spawnAcc: 0, elapsed: 0, flash: 0, flashColor: '#e5481c' }
    setStats({ tp: 0, fp: 0, fn: 0, tn: 0, score: 0 })
    setTime(GAME_SECONDS)
  }

  useEffect(() => {
    const cv = canvas.current!
    const ctx = cv.getContext('2d')!
    let raf = 0, last = performance.now()
    const resize = () => {
      const r = cv.getBoundingClientRect()
      const dpr = Math.min(2, window.devicePixelRatio || 1)
      cv.width = r.width * dpr; cv.height = r.height * dpr
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    }
    resize()
    const ro = new ResizeObserver(resize)
    ro.observe(cv)

    const frame = (now: number) => {
      const dt = Math.min(0.05, (now - last) / 1000)
      last = now
      const { width: w, height: h } = cv.getBoundingClientRect()
      const g = game.current
      const gate = w * 0.55
      const server = w - 46

      if (running) {
        g.elapsed += dt
        const rate = 5.5 + g.elapsed * 0.15          // packets per second, rising
        g.spawnAcc += dt * rate
        while (g.spawnAcc >= 1) {
          g.spawnAcc -= 1
          const [p, cls] = C.firewall.flows[g.cursor++ % C.firewall.flows.length]
          g.packets.push({ p, cls, x: 20, y: 24 + Math.random() * (h - 48), decided: false, blocked: false, vy: 0, alpha: 1 })
        }
        const speed = (w * 0.32) * (1 + g.elapsed / 60)
        const delta = { tp: 0, fp: 0, fn: 0, tn: 0 }
        for (const pk of g.packets) {
          if (pk.blocked) { pk.vy -= 260 * dt; pk.y += pk.vy * dt; pk.alpha -= dt * 1.4; continue }
          pk.x += speed * dt
          const attack = ORDER[pk.cls] !== 'normal'
          if (!pk.decided && pk.x >= gate) {
            pk.decided = true
            if (pk.p >= tRef.current) {
              pk.blocked = true
              pk.vy = -40
              if (attack) delta.tp++
              else delta.fp++
            }
          }
          if (pk.x >= server && !pk.blocked) {
            pk.alpha = 0
            if (attack) { delta.fn++; g.flash = 1; g.flashColor = '#e5481c' } else delta.tn++
          }
        }
        g.packets = g.packets.filter((pk) => pk.alpha > 0)
        if (delta.tp + delta.fp + delta.fn + delta.tn) {
          setStats((s) => ({
            tp: s.tp + delta.tp, fp: s.fp + delta.fp, fn: s.fn + delta.fn, tn: s.tn + delta.tn,
            score: s.score + delta.tp * SCORE.tp + delta.fp * SCORE.fp + delta.fn * SCORE.fn + delta.tn * SCORE.tn,
          }))
          if (delta.fp) { g.flash = Math.max(g.flash, 0.6); g.flashColor = '#c98a00' }
        }
      }
      g.flash = Math.max(0, g.flash - dt * 2)

      // draw
      ctx.clearRect(0, 0, w, h)
      ctx.fillStyle = 'rgba(239,234,224,0.05)'
      for (let gx = 0; gx < w; gx += 32) ctx.fillRect(gx, 0, 1, h)
      ctx.strokeStyle = 'rgba(239,234,224,0.35)'
      ctx.setLineDash([4, 6]); ctx.beginPath(); ctx.moveTo(gate, 8); ctx.lineTo(gate, h - 8); ctx.stroke(); ctx.setLineDash([])
      ctx.fillStyle = `rgba(${g.flashColor === '#e5481c' ? '229,72,28' : '201,138,0'},${0.15 + g.flash * 0.6})`
      ctx.fillRect(server, 10, 34, h - 20)
      ctx.fillStyle = 'rgba(239,234,224,0.8)'
      ctx.font = '10px JetBrains Mono'
      ctx.fillText('INTERNET', 10, 14)
      ctx.fillText(`FIREWALL ≥ ${tRef.current.toFixed(2)}`, gate + 6, 14)
      ctx.save(); ctx.translate(server + 21, h / 2); ctx.rotate(Math.PI / 2); ctx.textAlign = 'center'; ctx.fillText('SERVER', 0, 0); ctx.restore()
      for (const pk of g.packets) {
        ctx.globalAlpha = Math.max(0, pk.alpha)
        // unknown until it reaches the firewall, then shown in its true colour
        ctx.fillStyle = pk.decided ? TON_COLOR[ORDER[pk.cls]] === '#16140f' ? '#efeae0' : TON_COLOR[ORDER[pk.cls]] : 'rgba(239,234,224,0.55)'
        ctx.beginPath(); ctx.arc(pk.x, pk.y, pk.decided ? 5 : 4, 0, Math.PI * 2); ctx.fill()
        if (pk.blocked) { ctx.strokeStyle = '#efeae0'; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(pk.x, pk.y, 9, 0, Math.PI * 2); ctx.stroke() }
      }
      ctx.globalAlpha = 1
      raf = requestAnimationFrame(frame)
    }
    raf = requestAnimationFrame(frame)
    return () => { cancelAnimationFrame(raf); ro.disconnect() }
  }, [running])

  // round clock
  useEffect(() => {
    if (!running) return
    if (time <= 0) { setRunning(false); return }
    const t = window.setTimeout(() => setTime((s) => s - 1), 1000)
    return () => window.clearTimeout(t)
  }, [running, time])

  const { tp, fp, fn, tn, score } = stats
  const precision = tp + fp ? tp / (tp + fp) : 0
  const recall = tp + fn ? tp / (tp + fn) : 0
  const over = !running && time <= 0

  return (
    <div className="screen grain grid content-start gap-5 rounded-2xl p-5 md:p-7">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="grid gap-1">
          <span className="eyebrow text-red">Game 2 · Firewall defender</span>
          <h3 className="display text-[1.4rem] leading-tight">Set the threshold</h3>
        </div>
        <div className="flex items-baseline gap-4">
          <span className="num text-3xl text-ink">{score}</span>
          <span className="num text-sm text-ink-dim">{time}s</span>
        </div>
      </div>

      <canvas ref={canvas} className="h-[260px] w-full rounded-xl border border-hairline-strong" role="img"
        aria-label="Packets flow from the internet towards the server; the firewall blocks those whose attack probability is above the threshold" />

      <div className="grid gap-5 md:grid-cols-[minmax(0,1fr)_260px]">
        <div className="grid content-start gap-4">
          <div className="grid gap-2">
            <div className="flex items-baseline justify-between gap-3">
              <label htmlFor="fw-threshold" className="eyebrow">Block if P(attack) ≥</label>
              <output htmlFor="fw-threshold" className="num text-xl">{threshold.toFixed(2)}</output>
            </div>
            <Slider id="fw-threshold" value={[threshold]} min={0.05} max={0.95} step={0.05} onValueChange={([v]) => setThreshold(v)} aria-label="Blocking threshold" />
          </div>
          <div className="flex flex-wrap gap-2">
            <button type="button" onClick={() => { if (over || time === GAME_SECONDS) reset(); setRunning((r) => !r) }}
              className="rounded-full bg-red px-5 py-2 text-sm font-semibold text-paper transition hover:brightness-110">
              {running ? 'Pause' : over ? 'Play again' : time < GAME_SECONDS ? 'Resume' : 'Start'}
            </button>
            {!running && time < GAME_SECONDS && !over && (
              <button type="button" onClick={reset} className="rounded-full border border-hairline-strong px-4 py-2 text-sm">Reset</button>
            )}
          </div>
          <div className="num grid grid-cols-4 gap-2 text-center text-xs">
            {[['blocked attacks', tp, '+10', 'text-phosphor'], ['false alarms', fp, '−15', 'text-amber'], ['breaches', fn, '−25', 'text-red'], ['normal passed', tn, '+1', 'text-ink-muted']].map(([k, v, s, col]) => (
              <div key={k as string} className="grid gap-0.5 rounded-lg border border-hairline-strong px-2 py-2">
                <span className={cn('text-lg', col as string)}>{v}</span>
                <span className="text-[0.6rem] uppercase tracking-wider text-ink-dim">{k}</span>
                <span className="text-[0.6rem] text-ink-dim">{s}</span>
              </div>
            ))}
          </div>
          {over && (
            <p className="text-sm leading-relaxed text-ink-muted">
              Final score <b className="text-ink">{score}</b>: precision {pct(precision, 1)}, recall {pct(recall, 1)}. A low threshold catches more attacks but blocks real users;
              a high one lets more attacks through. Try again with a different setting.
            </p>
          )}
        </div>
        <div className="grid content-start gap-2">
          <span className="eyebrow">Trade-off on this stream</span>
          <TradeOff t={threshold} />
          <span className="text-xs text-ink-dim"><span className="text-phosphor">precision</span> · <span className="text-red">recall</span> · live: {pct(precision)} / {pct(recall)}</span>
        </div>
      </div>
      <p className="text-xs leading-relaxed text-ink-dim">
        Every packet is a real flow from the unseen test machines, carrying the final model's real attack probability.
        {C.firewall.oversampled
          ? `The model is unsure about only ${pct(C.firewall.unsureShareTest, 1)} of test flows, so this stream over-samples those hard cases (${pct(C.firewall.hardShare)} of packets) to make the threshold matter.`
          : `The model is unsure (attack probability between 2% and 98%) about ${pct(C.firewall.unsureShareTest)} of these flows, so where you set the threshold really matters.`}
      </p>
    </div>
  )
}

export function Games() {
  return (
    <Chapter
      id="play"
      index="06"
      tone="red"
      eyebrow="Your turn"
      title={<>Play against<br />the model</>}
      deck={<>Two games built on the project's real results. First, tune Gradient Boosting yourself and see why cross-validation alone can mislead. Then run the firewall with the final model's real attack probabilities and feel the trade-off between catching attacks and blocking real users. To type in your own flow, open the live models.</>}
    >
      <div className="grid gap-4 xl:grid-cols-2">
        <TuneGame />
        <Firewall />
      </div>
    </Chapter>
  )
}
