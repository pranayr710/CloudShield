import { useState } from 'react'
import { cn } from '@/lib/utils'
import { STREAMLIT_URL } from './space'

type Track = 'classify' | 'cluster' | 'regress'

const TRACKS: { key: Track; title: string; dataset: string; model: string; blurb: string }[] = [
  { key: 'classify', title: 'Classify a flow', dataset: 'TON_IoT', model: 'Bagging (tuned)', blurb: 'Normal traffic, or which of nine attacks?' },
  { key: 'cluster', title: 'Find its cluster', dataset: 'TON_IoT', model: 'K-Means', blurb: 'Which behaviour group does it fall into?' },
  { key: 'regress', title: 'Forecast reply size', dataset: 'UNSW-NB15', model: 'Random Forest', blurb: 'How many bytes will come back?' },
]

/** The live-model window: the deployed Streamlit app, embedded under the site's own header. */
export default function StreamlitWindow() {
  const [track, setTrack] = useState<Track>(() => {
    const t = new URLSearchParams(location.search).get('track')
    return t === 'cluster' || t === 'regress' ? t : 'classify'
  })
  const [loaded, setLoaded] = useState(false)
  const src = `${STREAMLIT_URL}/?track=${track}&embed=true&embed_options=light_theme`

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
          <a href={`${STREAMLIT_URL}/?track=${track}`} target="_blank" rel="noreferrer" className="text-xs underline decoration-red underline-offset-4">
            Open the app on its own ↗
          </a>
        </div>
      </header>

      <main className="gutter mx-auto grid max-w-[1360px] gap-8 py-10 md:py-14">
        <div className="grid gap-4 md:grid-cols-[minmax(0,7fr)_minmax(0,5fr)] md:items-end">
          <h1 className="display text-[clamp(2rem,5vw,3.8rem)] leading-[1.05]">Ask the<br />trained models</h1>
          <p className="max-w-[46ch] text-ink-muted md:justify-self-end">
            The three final models run in a Streamlit app. Describe a network flow, or start from an example, and the deployed model answers.
          </p>
        </div>

        <div className="grid gap-3 md:grid-cols-3" role="tablist">
          {TRACKS.map((x) => (
            <button key={x.key} type="button" role="tab" aria-selected={track === x.key} onClick={() => { setTrack(x.key); setLoaded(false) }}
              className={cn('grain grid gap-1 rounded-2xl border p-5 text-left transition', track === x.key ? 'border-ink bg-panel shadow-[0_14px_30px_rgba(22,20,15,0.10)]' : 'border-hairline-strong hover:border-ink')}>
              <span className="eyebrow">{x.dataset} · {x.model}</span>
              <span className="text-lg font-semibold">{x.title}</span>
              <span className="text-sm text-ink-muted">{x.blurb}</span>
            </button>
          ))}
        </div>

        <div className="relative overflow-hidden rounded-2xl border border-hairline-strong bg-panel">
          {!loaded && (
            <div className="absolute inset-0 grid place-items-center p-6 text-center">
              <p className="num text-sm text-ink-muted">Loading the live app<span className="caret">_</span><br />
                <span className="text-ink-dim">A sleeping Streamlit app can take up to a minute to wake; if it asks, press its wake-up button.</span></p>
            </div>
          )}
          <iframe key={track} src={src} title={`CloudShield live models: ${TRACKS.find((x) => x.key === track)!.title}`}
            onLoad={() => setLoaded(true)} className="relative block h-[1050px] w-full border-0 md:h-[880px]" allow="clipboard-write" />
        </div>
      </main>
    </div>
  )
}
