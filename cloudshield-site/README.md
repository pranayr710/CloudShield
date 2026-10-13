# CloudShield site

Interactive demo for the CloudShield ML project. Every number on the site comes from the project's notebooks and saved models.

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

## Updating the numbers (new dataset for Review 2)

The site reads one file, `src/data/cloudshield.json`. A script writes it from the ML project:

```bash
py -3.13 scripts/export_data.py
```

Run it with the Python whose scikit-learn version saved the models (currently 1.8). To switch datasets:

1. Rerun the notebooks so `reports/*.csv`, `processed_data/*.csv` and `models/*.joblib` are up to date.
2. Edit the `CONFIG` block at the top of `scripts/export_data.py`: file paths, label and target column names, the what-if feature, the leaky columns, and the few numbers that exist only in notebook output (`REG_CV`, `REG_BASELINE_RMSE`, `CONTAMINATION`).
3. Run the script, then run `npm run build`.

Text that is specific to UNSW-NB15 also needs editing by hand:
- the class descriptions in `src/lib/data.ts` (`CLASS_NOTE`, `CLASS_COLOR`)
- the dataset deck in `src/components/sections/dataset.tsx`
- the leakage card in `src/components/sections/traps.tsx` (`dmean × dpkts`)

The clustering numbers are currently computed by the export script on a 6,000-flow sample, because notebook 04 doesn't exist yet. Once it does, swap in its results.

## Stack

React 19 + TypeScript + Vite · Tailwind CSS 4 · shadcn/ui (`src/components/ui`) · GSAP 3.13 (ScrollTrigger, SplitText, ScrambleText).

| Path | Contents |
|---|---|
| `src/components/ui/hero-scrub.tsx` | Scroll-scrubbed hero (140 WebP frames in `public/frames/server`) |
| `src/components/sections/` | One file per chapter: dataset, classifier lab, confusion matrix, forecaster, models, traps, clusters, findings |
| `src/lib/data.ts` | Types and formatting helpers for the exported JSON |
| `scripts/export_data.py` | Builds `src/data/cloudshield.json` |
| `public/videos/` | Vecteezy clips re-encoded to 1280px H.264, credited in the footer |
