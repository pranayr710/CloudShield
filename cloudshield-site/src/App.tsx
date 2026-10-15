import { Nav } from '@/components/nav'
import { Ticker } from '@/components/ticker'
import { Hero } from '@/components/sections/hero'
import { StatsBand } from '@/components/sections/stats-band'
import { Dataset } from '@/components/sections/dataset'
import { Traps } from '@/components/sections/traps'
import { Interlude } from '@/components/sections/interlude'
import { ClassifierLab } from '@/components/sections/classifier-lab'
import { Leaderboard } from '@/components/sections/leaderboard'
import { Confusion } from '@/components/sections/confusion'
import { Clusters } from '@/components/sections/clusters'
import { Compare } from '@/components/sections/compare'
import { Games } from '@/components/sections/games'
import { Forecaster } from '@/components/sections/forecaster'
import { Models } from '@/components/sections/models'
import { Findings } from '@/components/sections/findings'
import { Footer } from '@/components/sections/footer'
import { D, T, int, pct } from '@/lib/data'

const mitm = T.overview.classes.find((c) => c.name === 'mitm')!

export default function App() {
  return (
    <>
      <Nav />
      <main>
        <Hero />
        <Ticker />
        <StatsBand />
        <Dataset />
        <Traps />
        <Interlude
          video="glitch"
          tone="red"
          quote={<>The rarest attack in the data appears {int(mitm.count)} times.</>}
          value={pct(mitm.count / T.overview.rowsClean, 2)}
          caption="of all flows are man-in-the-middle. Finding them is the hard part."
        />
        <ClassifierLab />
        <Leaderboard />
        <Confusion />
        <Games />
        <Clusters />
        <Compare />
        <Interlude
          video="switch"
          quote={<>Third track: every connection sends something back. How much?</>}
          value={D.regression.models[0].r2.toFixed(4)}
          caption={`R² of the ${D.regression.best} on ${int(D.classification.testRows)} unseen ${D.overview.dataset.name} flows`}
        />
        <Forecaster />
        <Models />
        <Findings />
      </main>
      <Footer />
    </>
  )
}
