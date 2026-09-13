"""Fetch UNSW-NB15 partition CSVs into data/raw/UNSW-NB15/.

    python data/download_unsw_nb15.py

CloudShield Track 2 (classification). Verifies the two partition files against
their ACTUAL shapes.

The official distribution ships the two files with their names swapped:
    UNSW_NB15_training-set.csv  ->   82,332 rows   (the paper says 175,341)
    UNSW_NB15_testing-set.csv   ->  175,341 rows   (the paper says  82,332)

The expectations below encode the real contents, not the published ones, so a
correct download does not fail verification. It does not affect modelling: the
pipeline concatenates both files and builds its own stratified split.
"""

import sys
from pathlib import Path

DEST = Path(__file__).resolve().parent / "raw" / "UNSW-NB15"
KAGGLE_SLUG = "mrwellsdavid/unsw-nb15"

EXPECTED = {
    "UNSW_NB15_training-set.csv": (82_332, 45),
    "UNSW_NB15_testing-set.csv": (175_341, 45),
}

INSTRUCTIONS = f"""
{'=' * 72}
UNSW-NB15 - DOWNLOAD REQUIRED  (CloudShield Track 2, classification)
{'=' * 72}

Destination: {DEST}

Option 1 - Kaggle (fastest)
    https://www.kaggle.com/datasets/{KAGGLE_SLUG}/data

    Or with the Kaggle CLI configured (~/.kaggle/kaggle.json):

        kaggle datasets download -d {KAGGLE_SLUG} -p data/raw/UNSW-NB15 --unzip

Option 2 - Official UNSW page (authoritative; cite this one)
    https://research.unsw.edu.au/projects/unsw-nb15-dataset
    -> UNSW SharePoint -> CSV Files -> Training and Testing Sets
    (The cloudstor.aarnet.edu.au link in older tutorials is dead.)

Take ONLY these two files:
    UNSW_NB15_training-set.csv   (~15 MB,  82,332 rows x 45 cols)
    UNSW_NB15_testing-set.csv    (~31 MB, 175,341 rows x 45 cols)

Do NOT use UNSW-NB15_1.csv .. _4.csv - that is the 2.54M-row raw dump with
49 columns, no header row, and blank cells instead of Normal in attack_cat.

CITATION
    Moustafa, N. and Slay, J. (2015). UNSW-NB15: a comprehensive data set for
    network intrusion detection systems. MilCIS.
{'=' * 72}
"""


def verify() -> bool:
    import pandas as pd

    ok = True
    for name, (rows, cols) in EXPECTED.items():
        path = DEST / name
        if not path.exists():
            print(f"  MISSING  {name}")
            ok = False
            continue
        n_cols = len(pd.read_csv(path, nrows=0).columns)
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            n_rows = sum(1 for _ in fh) - 1
        if (n_rows, n_cols) != (rows, cols):
            print(f"  MISMATCH {name}: expected {rows:,}x{cols}, "
                  f"got {n_rows:,}x{n_cols}")
            ok = False
        else:
            print(f"  OK       {name}  ({n_rows:,} x {n_cols})")
    return ok


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    if all((DEST / f).exists() for f in EXPECTED):
        print("Partition files present. Verifying ...")
        if verify():
            print("\nVerified. Next: python -m src.preprocessing.unsw_nb15")
            return 0
        print("\nVerification failed - the files do not match expected shapes.")
    print(INSTRUCTIONS)
    return 1


if __name__ == "__main__":
    sys.exit(main())
