import { D, T } from '@/lib/data'

/** Edit this list to the project team. */
const TEAM = [
  ['R. Meghapranay', 'CB.SC.U4CSE24440'],
  ['P. Krishna Chaitanaya', 'CB.SC.U4CSES24436'],
  ['M. Rishit Kumar', 'CB.SC.U4CSE24428'],
  ['K. Aditya', 'CB.SC.U4CSE24423'],
] as const

const unsw = D.overview.dataset
const ton = T.overview
const link = 'text-ink underline decoration-red underline-offset-4'

export function Footer() {
  return (
    <footer className="gutter border-t border-hairline-strong pb-12 pt-20">
      <div className="mx-auto grid max-w-[1360px] gap-14">
        <p className="display text-[clamp(2.2rem,6.5vw,5.5rem)] leading-[0.9]">
          Cloud<span className="text-red">Shield</span>
        </p>
        <div className="grid gap-10 md:grid-cols-3">
          <div className="grid content-start gap-3">
            <span className="eyebrow">Team</span>
            <ul className="grid gap-2">
              {TEAM.map(([name, reg]) => (
                <li key={reg} className="flex flex-wrap items-baseline justify-between gap-x-4 border-b border-hairline-strong pb-2">
                  <span>{name}</span><span className="num text-xs text-ink-dim">{reg}</span>
                </li>
              ))}
            </ul>
          </div>
          <div className="grid content-start gap-3 text-sm leading-relaxed text-ink-muted">
            <span className="eyebrow">Project</span>
            <p>23CSE301 Machine Learning, capstone project. Department of Computer Science and Engineering.</p>
            <p>Classification and clustering on {ton.name}; regression on {unsw.name}. Every number on this site is exported from the project notebooks and saved models.</p>
          </div>
          <div className="grid content-start gap-3 text-sm leading-relaxed text-ink-muted">
            <span className="eyebrow">Sources</span>
            <p>{ton.name}: {ton.citation} <a className={link} href={ton.url} target="_blank" rel="noreferrer">Dataset page</a>.</p>
            <p>{unsw.name}: {unsw.citation} <a className={link} href={unsw.url} target="_blank" rel="noreferrer">Dataset page</a>.</p>
            <p>Videos by <a className={link} href="https://www.vecteezy.com/free-videos" target="_blank" rel="noreferrer">Vecteezy</a>, used under its free licence.</p>
          </div>
        </div>
        <p className="text-xs text-ink-dim">An independent student project. Attack traffic in both datasets was generated on research testbeds.</p>
      </div>
    </footer>
  )
}
