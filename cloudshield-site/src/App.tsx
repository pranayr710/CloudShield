import { Nav } from '@/components/nav'
import { Ticker } from '@/components/ticker'
import { Hero } from '@/components/sections/hero'
import { StatsBand } from '@/components/sections/stats-band'
import { Dataset } from '@/components/sections/dataset'
import { Interlude } from '@/components/sections/interlude'
import { ClassifierLab } from '@/components/sections/classifier-lab'
import { Confusion } from '@/components/sections/confusion'
import { Forecaster } from '@/components/sections/forecaster'
import { Models } from '@/components/sections/models'
import { Traps } from '@/components/sections/traps'
import { Clusters } from '@/components/sections/clusters'
import { Findings } from '@/components/sections/findings'
import { Footer } from '@/components/sections/footer'
import { D, int, pct } from '@/lib/data'

const { overview: O, classification: C, regression: R } = D
const worms = O.classCounts.find((c) => c.name === 'Worms')!.count

export default function App() {
  return (
    <>
      <Nav />
      <main>
        <Hero />
        <Ticker />
        <StatsBand />
        <Dataset />
        <Interlude
          video="glitch"
          tone="red"
          quote={<>The rarest attack in the data appears {int(worms)} times.</>}
          value={pct(worms / O.rows, 2)}
          caption="of all flows are Worms. Finding them is the hard part."
        />
        <ClassifierLab />
        <Confusion />
        <Interlude
          video="switch"
          quote="Every connection sends something back. How much?"
          value={R.models[0].r2.toFixed(4)}
          caption={`R² of the ${R.best} on ${int(C.testRows)} unseen flows`}
        />
        <Forecaster />
        <Models />
        <Traps />
        <Clusters />
        <Findings />
      </main>
      <Footer />
    </>
  )
}
