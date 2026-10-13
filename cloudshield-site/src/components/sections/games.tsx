import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Chapter } from '@/components/chapter'
import { Slider } from '@/components/ui/slider'
import { T, TON_COLOR, TON_NOTE, bytes, int, pct, tonLabel, type TonFlow } from '@/lib/data'
import { cn } from '@/lib/utils'

const C = T.classification
const ORDER = C.order
const ROUNDS = 10
const SECONDS = 12

/* ================================================================== Game 1: beat the model */

type Answer = { flow: TonFlow; guess: string | null }

function pickRounds(): TonFlow[] {
  const shuffled = [...C.quiz].sort(() => Math.random() - 0.5)
  const misses = shuffled.filter((f) => f.pred !== f.label).slice(0, 3)
  const rest = shuffled.filter((f) => !misses.includes(f))
  const seen = new Set<string>()
  const varied = rest.filter((f) => (seen.has(f.label) ? false : (seen.add(f.label), true)))
  return [...misses, ...varied, ...rest].slice(0, ROUNDS).sort(() => Math.random() - 0.5)
}

function Stat({ k, v }: { k: string; v: string }) {
  return (
    <div className="grid gap-0.5 rounded-lg bg-paper px-3 py-2">
      <span className="text-[0.65rem] uppercase tracking-wider text-ink-dim">{k}</span>
      <span className="num text-sm">{v}</span>
    </div>
  )
}

function BeatTheModel() {
  const [rounds, setRounds] = useState<TonFlow[]>([])
  const [i, setI] = useState(0)
  const [answers, setAnswers] = useState<Answer[]>([])
  const [left, setLeft] = useState(SECONDS)
  const [finished, setFinished] = useState(false)
  const phase: 'intro' | 'play' | 'reveal' | 'done' = rounds.length === 0 ? 'intro' : finished ? 'done' : answers.length > i ? 'reveal' : 'play'
  const flow = rounds[i]
  const current = answers[i]

  const start = () => { setRounds(pickRounds()); setI(0); setAnswers([]); setLeft(SECONDS); setFinished(false) }
  const answer = useCallback((guess: string | null) => {
    setAnswers((a) => (a.length > i ? a : [...a, { flow: rounds[i], guess }]))
  }, [i, rounds])
  const next = () => {
    if (i >= ROUNDS - 1) { setFinished(true); return }
    setI(i + 1); setLeft(SECONDS)
  }

  // countdown while a round is open
  useEffect(() => {
    if (phase !== 'play' || !flow || finished) return
    if (left <= 0) { answer(null); return }
    const t = window.setTimeout(() => setLeft((s) => s - 1), 1000)
    return () => window.clearTimeout(t)
  }, [phase, left, flow, answer, finished])

  // keys 1-9 and 0 pick a class, Enter moves on
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!flow || finished) return
      if (phase === 'play' && /^[0-9]$/.test(e.key)) answer(ORDER[(Number(e.key) + 9) % 10])
      else if (phase === 'reveal' && e.key === 'Enter') next()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  })

  const you = answers.filter((a) => a.guess === a.flow.label).length
  const model = answers.filter((a) => a.flow.pred === a.flow.label).length

  return (
    <div className="grain grid content-start gap-5 rounded-2xl border border-hairline-strong bg-panel p-5 md:p-7">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="grid gap-1">
          <span className="eyebrow">Game 1 · Beat the model</span>
          <h3 className="display text-[1.4rem] leading-tight">You vs {C.best}</h3>
        </div>
        <div className="flex items-center gap-2" aria-live="polite">
          <div className="grid justify-items-center rounded-xl bg-ink px-4 py-2 text-paper">
            <span className="num text-2xl leading-none">{you}</span><span className="text-[0.6rem] uppercase tracking-wider opacity-70">you</span>
          </div>
          <span className="num text-ink-dim">:</span>
          <div className="grid justify-items-center rounded-xl bg-red px-4 py-2 text-paper">
            <span className="num text-2xl leading-none">{model}</span><span className="text-[0.6rem] uppercase tracking-wider opacity-80">model</span>
          </div>
        </div>
      </div>

      {phase === 'intro' && (
        <div className="grid gap-4">
          <p className="max-w-[56ch] text-sm leading-relaxed text-ink-muted">
            {ROUNDS} real flows from the test set. Read the clues, pick the traffic type before the {SECONDS}-second timer runs out, then see what the model said.
            The model got some of these wrong, so it can be beaten. Keys 1 to 0 pick a class; Enter moves on.
          </p>
          <button type="button" onClick={start} className="justify-self-start rounded-full bg-ink px-5 py-2.5 text-sm font-semibold text-paper transition hover:bg-red">Start the round</button>
        </div>
      )}

      {flow && !finished && (
        <>
          <div className="flex items-center justify-between gap-3 text-xs text-ink-dim">
            <span className="num">flow {i + 1} of {ROUNDS} · #{flow.id}</span>
            <span className="num">{phase === 'play' ? `${left}s` : current?.guess ? 'answered' : 'time up'}</span>
          </div>
          <div className="h-1 overflow-hidden rounded-full bg-panel-2">
            <div className="h-full bg-red transition-[width] duration-1000 ease-linear" style={{ width: `${phase === 'play' ? (left / SECONDS) * 100 : 0}%` }} />
          </div>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            <Stat k="protocol" v={flow.f.proto} />
            <Stat k="service" v={flow.f.service === '-' ? 'none' : flow.f.service} />
            <Stat k="state" v={flow.f.conn_state} />
            <Stat k="dest port" v={String(flow.f.dst_port)} />
            <Stat k="duration" v={`${flow.f.duration.toFixed(3)} s`} />
            <Stat k="packets out / in" v={`${int(flow.f.src_pkts)} / ${int(flow.f.dst_pkts)}`} />
            <Stat k="bytes out" v={bytes(flow.f.src_bytes)} />
            <Stat k="bytes in" v={bytes(flow.f.dst_bytes)} />
          </div>

          <div className="grid grid-cols-2 gap-1.5 sm:grid-cols-5" role="group" aria-label="Your guess">
            {ORDER.map((c, k) => {
              const picked = current?.guess === c
              const truth = phase === 'reveal' && c === flow.label
              const modelSaid = phase === 'reveal' && c === flow.pred
              return (
                <button key={c} type="button" disabled={phase !== 'play'} onClick={() => answer(c)} title={TON_NOTE[c]}
                  className={cn('relative flex items-center gap-2 rounded-lg border px-2.5 py-2 text-left text-xs font-medium transition',
                    phase === 'play' ? 'border-hairline-strong bg-paper hover:border-ink' : 'border-transparent bg-paper/60',
                    truth && 'border-signal bg-signal/10', picked && !truth && 'border-red bg-red/10')}>
                  <span className="num text-[0.6rem] text-ink-dim">{(k + 1) % 10}</span>
                  <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: TON_COLOR[c] }} />
                  <span className="truncate">{tonLabel(c)}</span>
                  {modelSaid && <span className="absolute -top-2 right-1.5 rounded bg-red px-1 text-[0.55rem] leading-4 text-paper">model</span>}
                </button>
              )
            })}
          </div>

          {phase === 'reveal' && current && (
            <div className="grid gap-3 rounded-xl border border-hairline-strong bg-paper p-4 sm:grid-cols-[1fr_auto] sm:items-center">
              <div className="grid gap-1 text-sm">
                <p>Truth: <b style={{ color: TON_COLOR[flow.label] }}>{tonLabel(flow.label)}</b>. {TON_NOTE[flow.label]}</p>
                <p className="text-ink-muted">
                  You: {current.guess ? <b className={current.guess === flow.label ? 'text-signal' : 'text-red'}>{tonLabel(current.guess)}</b> : <b className="text-red">no answer</b>}
                  {' · '}Model: <b className={flow.pred === flow.label ? 'text-signal' : 'text-red'}>{tonLabel(flow.pred)}</b> ({pct(flow.proba[0]?.p ?? 0)} sure)
                </p>
              </div>
              <button type="button" onClick={next} className="rounded-full bg-ink px-4 py-2 text-sm font-semibold text-paper transition hover:bg-red">
                {i >= ROUNDS - 1 ? 'See the result' : 'Next flow →'}
              </button>
            </div>
          )}
        </>
      )}

      {finished && (
        <div className="grid justify-items-start gap-4">
          <span className={cn('stamp text-base', you > model ? 'text-signal' : you === model ? 'text-amber' : 'text-red')}>
            {you > model ? 'You beat the model' : you === model ? 'A draw' : 'The model wins'}
          </span>
          <p className="text-sm leading-relaxed text-ink-muted">
            You named {you} of {ROUNDS} flows, the model {model}. Across all {int(C.testRows)} test flows the model is right {pct(C.final.acc, 1)} of the time;
            this round was stacked with some of its rare mistakes.
          </p>
          <button type="button" onClick={start} className="rounded-full bg-ink px-5 py-2.5 text-sm font-semibold text-paper transition hover:bg-red">Play again</button>
        </div>
      )}
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
        Every packet is a real test flow carrying the final model's real attack probability. The model is unsure about only {pct(C.firewall.unsureShareTest, 1)} of all test flows,
        so this stream over-samples those hard cases ({pct(C.firewall.hardShare)} of packets) to make the threshold matter.
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
      deck={<>Two games built on real test flows and the final model's real outputs. First, try to name attacks faster and better than {C.best}. Then run the firewall yourself and feel the trade-off between catching attacks and blocking real users.</>}
    >
      <div className="grid gap-4 xl:grid-cols-2">
        <BeatTheModel />
        <Firewall />
      </div>
    </Chapter>
  )
}
