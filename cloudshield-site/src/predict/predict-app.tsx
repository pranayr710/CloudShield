import { useEffect, useRef, useState, type ReactNode } from 'react'
import { Client } from '@gradio/client'
import { TON_COLOR, bytes, pct, tonLabel } from '@/lib/data'
import { cn } from '@/lib/utils'
import { SPACE, SPACE_URL } from './space'


type Track = 'classify' | 'cluster' | 'regress'
type Status = 'connecting' | 'ready' | 'error'

type TonForm = { proto: string; service: string; conn_state: string; dst_port: number; duration: number; src_bytes: number; dst_bytes: number; src_pkts: number; dst_pkts: number }
type RegForm = { proto: string; service: string; state: string; dur: number; spkts: number; dpkts: number; sbytes: number; sttl: number; dttl: number; dloss: number }

type ClassifyOut = { prediction: string; label: string; confidence: number; note: string; probabilities: Record<string, number> }
type ClusterOut = { cluster: number; k: number; distance: number; distances: number[]; size: number; mix: Record<string, number> }
type RegressOut = { bytes: number; low: number; high: number }
type Result =
  | { track: 'classify'; out: ClassifyOut; input: TonForm; ms: number }
  | { track: 'cluster'; out: ClusterOut; input: TonForm; ms: number }
  | { track: 'regress'; out: RegressOut; input: RegForm; ms: number }

const TRACKS: { key: Track; title: string; dataset: string; model: string; blurb: string }[] = [
  { key: 'classify', title: 'Classify a flow', dataset: 'TON_IoT', model: 'Gradient Boosting (tuned)', blurb: 'Normal traffic, or which of nine attacks?' },
  { key: 'cluster', title: 'Find its cluster', dataset: 'TON_IoT', model: 'K-Means', blurb: 'Which behaviour group does it fall into?' },
  { key: 'regress', title: 'Forecast reply size', dataset: 'UNSW-NB15', model: 'Random Forest', blurb: 'How many bytes will come back?' },
]

const TON_PRESETS: { name: string; v: TonForm }[] = [
  { name: 'Web page load', v: { proto: 'tcp', service: 'http', conn_state: 'SF', dst_port: 80, duration: 0.22, src_bytes: 320, dst_bytes: 1920, src_pkts: 5, dst_pkts: 6 } },
  { name: 'DNS lookup', v: { proto: 'udp', service: 'dns', conn_state: 'SF', dst_port: 53, duration: 0.004, src_bytes: 74, dst_bytes: 90, src_pkts: 2, dst_pkts: 2 } },
  { name: 'Rejected probe', v: { proto: 'tcp', service: '-', conn_state: 'REJ', dst_port: 9594, duration: 0.003, src_bytes: 0, dst_bytes: 0, src_pkts: 2, dst_pkts: 1 } },
  { name: 'Unanswered SYN', v: { proto: 'tcp', service: '-', conn_state: 'S0', dst_port: 445, duration: 0, src_bytes: 0, dst_bytes: 0, src_pkts: 1, dst_pkts: 0 } },
  { name: 'FTP session', v: { proto: 'tcp', service: 'ftp', conn_state: 'SF', dst_port: 21, duration: 3.59, src_bytes: 86, dst_bytes: 215, src_pkts: 18, dst_pkts: 12 } },
  { name: 'TLS session', v: { proto: 'tcp', service: 'ssl', conn_state: 'SF', dst_port: 443, duration: 3.23, src_bytes: 1179, dst_bytes: 5861, src_pkts: 15, dst_pkts: 18 } },
]
const REG_PRESETS: { name: string; v: RegForm }[] = [
  { name: 'Small exchange', v: { proto: 'tcp', service: '-', state: 'FIN', dur: 0.45, spkts: 10, dpkts: 6, sbytes: 738, sttl: 254, dttl: 252, dloss: 0 } },
  { name: 'Busy download', v: { proto: 'tcp', service: '-', state: 'FIN', dur: 1.85, spkts: 28, dpkts: 72, sbytes: 1320, sttl: 62, dttl: 252, dloss: 2 } },
  { name: 'Large transfer', v: { proto: 'tcp', service: '-', state: 'FIN', dur: 4.2, spkts: 300, dpkts: 800, sbytes: 16000, sttl: 31, dttl: 29, dloss: 14 } },
  { name: 'One-way UDP', v: { proto: 'udp', service: 'dns', state: 'INT', dur: 0, spkts: 2, dpkts: 0, sbytes: 114, sttl: 254, dttl: 0, dloss: 0 } },
]

const PROTOS = ['tcp', 'udp', 'icmp']
const SERVICES = ['-', 'http', 'dns', 'ssl', 'ftp', 'smtp', 'ssh', 'dhcp', 'ftp-data']
const STATES = ['SF', 'S0', 'REJ', 'S1', 'S2', 'S3', 'OTH', 'RSTO', 'RSTR', 'RSTOS0', 'SH', 'SHR']
const U_PROTOS = ['tcp', 'udp', 'arp', 'unas', 'ospf', 'sctp']
const U_SERVICES = ['-', 'dns', 'http', 'ftp', 'ftp-data', 'smtp', 'ssh', 'pop3', 'snmp', 'ssl', 'dhcp', 'irc', 'radius']
const U_STATES = ['FIN', 'INT', 'CON', 'REQ', 'RST', 'ACC', 'CLO']
/** class colours that stay readable on the dark result screen */
const onScreen = (c: string) => (c === 'normal' ? '#5fd39a' : c === 'ransomware' ? '#efeae0' : c === 'backdoor' ? '#c9a27e' : TON_COLOR[c])
const CLUSTER_COLOR = ['#e5481c', '#2f5d9e', '#0f6e5a', '#c98a00', '#7b3fa0', '#c2185b', '#00838f', '#5b4636']

/* ------------------------------------------------------------------ form controls */
function Field({ label, children, hint }: { label: string; children: ReactNode; hint?: string }) {
  return (
    <label className="grid gap-1.5">
      <span className="eyebrow">{label}</span>
      {children}
      {hint && <span className="text-[0.7rem] text-ink-dim">{hint}</span>}
    </label>
  )
}
const inputCls = 'w-full rounded-lg border border-hairline-strong bg-paper px-3 py-2 text-sm outline-none transition focus:border-ink'

function Select({ value, options, onChange }: { value: string; options: string[]; onChange: (v: string) => void }) {
  return (
    <select value={value} onChange={(e) => onChange(e.target.value)} className={cn(inputCls, 'num')}>
      {options.map((o) => <option key={o} value={o}>{o === '-' ? '- (none)' : o}</option>)}
    </select>
  )
}
function Num({ value, onChange, step = 1, min = 0 }: { value: number; onChange: (v: number) => void; step?: number; min?: number }) {
  return <input type="number" value={value} min={min} step={step} onChange={(e) => onChange(Number(e.target.value))} className={cn(inputCls, 'num')} />
}

/* ------------------------------------------------------------------ results */
function ClassifyView({ out }: { out: ClassifyOut }) {
  const color = onScreen(out.prediction)
  return (
    <div className="grid gap-5">
      <div className="grid gap-1">
        <span className="eyebrow">{out.prediction === 'normal' ? 'Verdict' : 'Attack detected'}</span>
        <span className="display text-[clamp(2rem,6vw,3.6rem)] leading-none" style={{ color }}>{out.label}</span>
        <span className="num text-sm text-ink-muted">{pct(out.confidence, 1)} confidence</span>
        <p className="text-sm text-ink-muted">{out.note}</p>
      </div>
      <div className="grid gap-1.5">
        {Object.entries(out.probabilities).map(([c, p]) => (
          <div key={c} className="grid grid-cols-[6.5rem_minmax(0,1fr)_3.4rem] items-center gap-3 text-xs">
            <span className={c === out.prediction ? 'text-ink' : 'text-ink-muted'}>{tonLabel(c)}</span>
            <span className="relative h-2 overflow-hidden rounded-sm bg-panel-2">
              <span className="absolute inset-y-0 left-0 transition-[width] duration-700" style={{ width: `${p * 100}%`, background: onScreen(c) }} />
            </span>
            <span className="num text-right">{pct(p, p < 0.1 ? 1 : 0)}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

function ClusterView({ out }: { out: ClusterOut }) {
  const maxD = Math.max(...out.distances)
  return (
    <div className="grid gap-5">
      <div className="grid gap-1">
        <span className="eyebrow">Behaviour group</span>
        <span className="display text-[clamp(2rem,6vw,3.6rem)] leading-none" style={{ color: CLUSTER_COLOR[(out.cluster - 1) % CLUSTER_COLOR.length] }}>Cluster {out.cluster}</span>
        <span className="num text-sm text-ink-muted">of {out.k} · {out.size.toLocaleString('en-US')} training flows in this cluster</span>
      </div>
      <div className="grid gap-1.5">
        <span className="eyebrow">What that cluster held during training</span>
        {Object.entries(out.mix).map(([c, s]) => (
          <div key={c} className="grid grid-cols-[6.5rem_minmax(0,1fr)_3rem] items-center gap-3 text-xs">
            <span className="text-ink-muted">{tonLabel(c)}</span>
            <span className="relative h-2 rounded-sm bg-panel-2"><span className="absolute inset-y-0 left-0 rounded-sm" style={{ width: `${s * 100}%`, background: onScreen(c) }} /></span>
            <span className="num text-right">{pct(s)}</span>
          </div>
        ))}
      </div>
      <div className="grid gap-1.5">
        <span className="eyebrow">Distance to each cluster centre (shorter is closer)</span>
        {out.distances.map((d, i) => (
          <div key={i} className="grid grid-cols-[4.5rem_minmax(0,1fr)_3rem] items-center gap-3 text-xs">
            <span className={i + 1 === out.cluster ? 'text-ink' : 'text-ink-dim'}>Cluster {i + 1}</span>
            <span className="relative h-1.5 rounded-sm bg-panel-2"><span className="absolute inset-y-0 left-0 rounded-sm" style={{ width: `${(d / maxD) * 100}%`, background: i + 1 === out.cluster ? CLUSTER_COLOR[i % CLUSTER_COLOR.length] : 'rgba(239,234,224,0.35)' }} /></span>
            <span className="num text-right">{d.toFixed(1)}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

function RegressView({ out }: { out: RegressOut }) {
  const lg = (v: number) => Math.log10(v + 1) / 7
  return (
    <div className="grid gap-5">
      <div className="grid gap-1">
        <span className="eyebrow">Predicted reply size</span>
        <span className="display text-[clamp(2rem,6vw,3.6rem)] leading-none text-phosphor">{bytes(out.bytes)}</span>
        <span className="num text-sm text-ink-muted">trees agree on {bytes(out.low)} to {bytes(out.high)}</span>
      </div>
      <div className="grid gap-2">
        <div className="relative h-3 rounded-full bg-panel-2">
          <span className="absolute inset-y-0 rounded-full bg-phosphor/40" style={{ left: `${lg(out.low) * 100}%`, width: `${Math.max(1, (lg(out.high) - lg(out.low)) * 100)}%` }} />
          <span className="absolute top-1/2 h-5 w-1 -translate-y-1/2 rounded bg-phosphor" style={{ left: `${lg(out.bytes) * 100}%` }} />
        </div>
        <div className="num flex justify-between text-[0.65rem] text-ink-dim">{['0', '10 B', '100 B', '1 KB', '10 KB', '100 KB', '1 MB', '10 MB'].map((t) => <span key={t}>{t}</span>)}</div>
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------ page */
export default function PredictApp() {
  const [status, setStatus] = useState<Status>('connecting')
  const [track, setTrack] = useState<Track>('classify')
  const [ton, setTon] = useState<TonForm>(TON_PRESETS[0].v)
  const [reg, setReg] = useState<RegForm>(REG_PRESETS[1].v)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<Result | null>(null)
  const [history, setHistory] = useState<Result[]>([])
  const client = useRef<Promise<Client> | null>(null)

  const connect = () => {
    setStatus('connecting')
    client.current = Client.connect(SPACE)
    client.current.then(() => setStatus('ready')).catch(() => setStatus('error'))
  }
  useEffect(connect, [])

  const run = async () => {
    setBusy(true)
    setError(null)
    const t0 = performance.now()
    try {
      if (!client.current) connect()
      const c = await client.current!
      setStatus('ready')
      const res = await c.predict(`/${track}`, track === 'regress' ? { ...reg } : { ...ton })
      const data = res.data as unknown[]
      const out = data[data.length - 1]
      const r = { track, out, input: track === 'regress' ? reg : ton, ms: performance.now() - t0 } as Result
      setResult(r)
      setHistory((h) => [r, ...h].slice(0, 6))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setStatus('error')
    } finally {
      setBusy(false)
    }
  }

  const t = TRACKS.find((x) => x.key === track)!
  const setT = (k: keyof TonForm) => (v: string | number) => setTon((f) => ({ ...f, [k]: v }))
  const setR = (k: keyof RegForm) => (v: string | number) => setReg((f) => ({ ...f, [k]: v }))

  return (
    <div className="min-h-[100svh] bg-paper">
      <header className="border-b border-hairline-strong">
        <div className="gutter mx-auto flex h-14 max-w-[1360px] items-center justify-between gap-4">
          <a href="/" className="flex items-center gap-2.5" aria-label="Back to the CloudShield briefing">
            <svg viewBox="0 0 32 32" className="h-6 w-6" aria-hidden>
              <path d="M16 5 7 8.5v6.8c0 5.6 3.8 10 9 11.7 5.2-1.7 9-6.1 9-11.7V8.5z" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinejoin="round" />
              <circle cx="16" cy="14.5" r="2.6" className="fill-red" />
            </svg>
            <span className="display text-[0.85rem]">Cloud<span className="text-red">Shield</span></span>
            <span className="chip ml-2 hidden sm:inline">Live models</span>
          </a>
          <div className="flex items-center gap-3 text-xs">
            <span className="flex items-center gap-2">
              <span className={cn('h-2 w-2 rounded-full', status === 'ready' ? 'bg-signal' : status === 'error' ? 'bg-red' : 'animate-pulse bg-amber')} />
              <span className="text-ink-muted">{status === 'ready' ? 'Connected to Hugging Face' : status === 'error' ? 'Not connected' : 'Waking the Space…'}</span>
            </span>
            {status === 'error' && <button type="button" onClick={connect} className="underline decoration-red underline-offset-4">retry</button>}
            <a href={SPACE_URL} target="_blank" rel="noreferrer" className="hidden underline decoration-red underline-offset-4 md:inline">Open the Space ↗</a>
          </div>
        </div>
      </header>

      <main className="gutter mx-auto grid max-w-[1360px] gap-8 py-10 md:py-14">
        <div className="grid gap-4 md:grid-cols-[minmax(0,7fr)_minmax(0,5fr)] md:items-end">
          <h1 className="display text-[clamp(2rem,5vw,3.8rem)] leading-[1.05]">Ask the<br />trained models</h1>
          <p className="max-w-[46ch] text-ink-muted md:justify-self-end">
            Describe a network flow and send it to the models deployed on Hugging Face. The prediction comes back from the live app, the same one anyone can open on the Space.
          </p>
        </div>

        <div className="grid gap-3 md:grid-cols-3" role="tablist">
          {TRACKS.map((x) => (
            <button key={x.key} type="button" role="tab" aria-selected={track === x.key} onClick={() => { setTrack(x.key); setResult(null); setError(null) }}
              className={cn('grain grid gap-1 rounded-2xl border p-5 text-left transition', track === x.key ? 'border-ink bg-panel shadow-[0_14px_30px_rgba(22,20,15,0.10)]' : 'border-hairline-strong hover:border-ink')}>
              <span className="eyebrow">{x.dataset} · {x.model}</span>
              <span className="text-lg font-semibold">{x.title}</span>
              <span className="text-sm text-ink-muted">{x.blurb}</span>
            </button>
          ))}
        </div>

        <div className="grid gap-4 lg:grid-cols-[minmax(0,6fr)_minmax(0,6fr)]">
          <form className="grain grid content-start gap-5 rounded-2xl border border-hairline-strong bg-panel p-6 md:p-8" onSubmit={(e) => { e.preventDefault(); void run() }}>
            <div className="flex flex-wrap gap-2">
              <span className="eyebrow w-full">Start from an example</span>
              {(track === 'regress' ? REG_PRESETS : TON_PRESETS).map((p) => (
                <button key={p.name} type="button" className="chip transition hover:border-ink"
                  onClick={() => (track === 'regress' ? setReg(p.v as RegForm) : setTon(p.v as TonForm))}>{p.name}</button>
              ))}
            </div>
            {track !== 'regress' ? (
              <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
                <Field label="Protocol"><Select value={ton.proto} options={PROTOS} onChange={setT('proto')} /></Field>
                <Field label="Service"><Select value={ton.service} options={SERVICES} onChange={setT('service')} /></Field>
                <Field label="Connection state"><Select value={ton.conn_state} options={STATES} onChange={setT('conn_state')} /></Field>
                <Field label="Destination port"><Num value={ton.dst_port} onChange={setT('dst_port')} /></Field>
                <Field label="Duration (s)"><Num value={ton.duration} step={0.001} onChange={setT('duration')} /></Field>
                <span className="hidden sm:block" />
                <Field label="Bytes sent"><Num value={ton.src_bytes} onChange={setT('src_bytes')} /></Field>
                <Field label="Bytes received"><Num value={ton.dst_bytes} onChange={setT('dst_bytes')} /></Field>
                <span className="hidden sm:block" />
                <Field label="Packets sent"><Num value={ton.src_pkts} onChange={setT('src_pkts')} /></Field>
                <Field label="Packets received"><Num value={ton.dst_pkts} onChange={setT('dst_pkts')} /></Field>
              </div>
            ) : (
              <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
                <Field label="Protocol"><Select value={reg.proto} options={U_PROTOS} onChange={setR('proto')} /></Field>
                <Field label="Service"><Select value={reg.service} options={U_SERVICES} onChange={setR('service')} /></Field>
                <Field label="State"><Select value={reg.state} options={U_STATES} onChange={setR('state')} /></Field>
                <Field label="Duration (s)"><Num value={reg.dur} step={0.01} onChange={setR('dur')} /></Field>
                <Field label="Bytes sent"><Num value={reg.sbytes} onChange={setR('sbytes')} /></Field>
                <Field label="Packets sent"><Num value={reg.spkts} onChange={setR('spkts')} /></Field>
                <Field label="Packets received" hint="the strongest signal"><Num value={reg.dpkts} onChange={setR('dpkts')} /></Field>
                <Field label="Packets lost back"><Num value={reg.dloss} onChange={setR('dloss')} /></Field>
                <span className="hidden sm:block" />
                <Field label="Source TTL"><Num value={reg.sttl} onChange={setR('sttl')} /></Field>
                <Field label="Destination TTL"><Num value={reg.dttl} onChange={setR('dttl')} /></Field>
              </div>
            )}
            <p className="text-xs leading-relaxed text-ink-dim">
              {track === 'regress'
                ? 'Every other UNSW-NB15 column is set to its typical value from the training data.'
                : 'Every other TON_IoT column (DNS, SSL and HTTP details) is set to its "not applicable" value; IP-level byte counts add a 40-byte header per packet.'}
            </p>
            <button type="submit" disabled={busy} className="justify-self-start rounded-full bg-ink px-6 py-3 text-sm font-semibold text-paper transition hover:bg-red disabled:opacity-60">
              {busy ? 'Asking the model…' : `${t.title} →`}
            </button>
          </form>

          <section className="screen scanlines relative grid min-h-[420px] content-start gap-5 overflow-hidden rounded-2xl p-6 md:p-8" aria-live="polite">
            <div className="flex items-center justify-between gap-3">
              <span className="eyebrow text-red">{t.model} · {t.dataset}</span>
              {result && <span className="num text-xs text-ink-dim">answered in {(result.ms / 1000).toFixed(2)} s</span>}
            </div>
            {busy && <p className="num text-sm text-ink-muted">Sending the flow to Hugging Face<span className="caret">_</span><br /><span className="text-ink-dim">A sleeping Space can take up to a minute to wake.</span></p>}
            {error && !busy && <p className="text-sm text-red">The Space did not answer: {error}. Check the connection or open the Space directly.</p>}
            {!busy && !error && !result && <p className="text-sm text-ink-muted">Fill in a flow, or start from an example, and send it. The answer appears here.</p>}
            {!busy && result?.track === 'classify' && track === 'classify' && <ClassifyView out={result.out} />}
            {!busy && result?.track === 'cluster' && track === 'cluster' && <ClusterView out={result.out} />}
            {!busy && result?.track === 'regress' && track === 'regress' && <RegressView out={result.out} />}
          </section>
        </div>

        {history.length > 0 && (
          <div className="grid gap-3">
            <span className="eyebrow">This session</span>
            <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
              {history.map((h, i) => (
                <div key={i} className="grain flex items-center justify-between gap-3 rounded-xl border border-hairline-strong bg-panel px-4 py-3 text-sm">
                  <span className="text-ink-muted">{TRACKS.find((x) => x.key === h.track)!.title}</span>
                  <span className="num font-medium">
                    {h.track === 'classify' ? `${h.out.label} · ${pct(h.out.confidence)}` : h.track === 'cluster' ? `Cluster ${h.out.cluster}` : bytes(h.out.bytes)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  )
}
