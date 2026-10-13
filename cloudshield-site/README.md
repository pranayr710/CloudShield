# CloudShield site

Presentation site for the CloudShield ML project: classification and clustering on TON_IoT, regression on UNSW-NB15. Every number on the site comes from the project's notebooks and saved models.

## Run it

```bash
npm install          # first time only
npm run dev          # http://localhost:5173
```

For the review, use the production build. It runs more smoothly:

```bash
npm run build
npm run preview      # http://localhost:4173
```

The videos and hero frames are stored locally, so the only things that need internet are the Google fonts.

## Updating the numbers

The site reads two files, written by two scripts from the ML project's notebooks and saved models:

| File | Script | Content |
|---|---|---|
| `src/data/toniot.json` | `scripts/export_ton_iot.py` | TON_IoT classification (10 models, final model, confusion matrix, stress test) and clustering |
| `src/data/cloudshield.json` | `scripts/export_data.py` | UNSW-NB15 regression |

After rerunning the notebooks:

```bash
py -3.10 scripts/export_ton_iot.py     # the Python that ran notebooks/classification.ipynb
py -3.13 scripts/export_data.py        # the Python that saved the UNSW-NB15 models
npm run build
```

Each script must run with the scikit-learn version that saved its models, or joblib will refuse to load them.

## Stack

React 19 + TypeScript + Vite · Tailwind CSS 4 · shadcn/ui (`src/components/ui`) · GSAP 3.13 (ScrollTrigger, SplitText, ScrambleText).

| Path | Contents |
|---|---|
| `src/components/ui/hero-scrub.tsx` | Scroll-scrubbed hero (140 WebP frames in `public/frames/server`) |
| `src/components/sections/` | One file per chapter: dataset, traps, classifier lab, ten-model leaderboard, errors and stress test, clusters, regression forecaster and models, findings |
| `src/lib/data.ts` | Types and formatting helpers for the exported JSON |
| `scripts/export_ton_iot.py`, `scripts/export_data.py` | Build the two data files |
| `public/videos/` | Vecteezy clips re-encoded to 1280px H.264, credited in the footer |
