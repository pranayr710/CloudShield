"""Fetch CSE-CIC-IDS2018 processed CSVs into data/raw/CSE-CIC-IDS2018/.

    python data/download_cse_cic_ids2018.py            # show instructions
    python data/download_cse_cic_ids2018.py --aws      # sync from AWS Open Data

CloudShield Track 1 (regression). The full set is ~6.5 GB across 10 daily
capture files (~16M flows), one file per attack day. Which files the project
actually uses must be documented in the notebook (spec step 7).

This script does not guess column names. Run the inspection step after
downloading.
"""

import argparse
import subprocess
import sys
from pathlib import Path

DEST = Path(__file__).resolve().parent / "raw" / "CSE-CIC-IDS2018"
S3_URI = "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/"

INSTRUCTIONS = f"""
{'=' * 72}
CSE-CIC-IDS2018 - DOWNLOAD REQUIRED  (CloudShield Track 1, regression)
{'=' * 72}

Destination: {DEST}

Option 1 - AWS Open Data Registry (official, no credentials needed)
    Install the AWS CLI, then re-run this script with --aws, or run directly:

        aws s3 sync --no-sign-request "{S3_URI}" "{DEST}"

    The full sync is ~6.5 GB. To start with a single day instead:

        aws s3 cp --no-sign-request \\
          "{S3_URI}Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv" "{DEST}"

Option 2 - Official CIC page
    https://www.unb.ca/cic/datasets/ids-2018.html

Option 3 - Kaggle mirrors
    Search "CSE-CIC-IDS2018" on kaggle.com/datasets and take the
    "Processed Traffic Data for ML Algorithms" CSVs.

AFTER DOWNLOADING
    Do not assume the byte-rate column is named fl_byt_s. Run:

        python -m src.preprocessing.cse_cic_ids2018

    It prints the real schema of every file present and reports which
    byte-rate alias was found. The pipeline refuses to run until a known
    alias resolves, rather than substituting a different column.

CITATION
    Sharafaldin, I., Habibi Lashkari, A. and Ghorbani, A. (2018). Toward
    Generating a New Intrusion Detection Dataset and Intrusion Traffic
    Characterization. ICISSP.
{'=' * 72}
"""


def sync_from_aws() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    cmd = ["aws", "s3", "sync", "--no-sign-request", S3_URI, str(DEST)]
    print("running:", " ".join(cmd))
    try:
        return subprocess.call(cmd)
    except FileNotFoundError:
        print("\nAWS CLI not found. Install it, or use Option 2 or 3 below.")
        print(INSTRUCTIONS)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch CSE-CIC-IDS2018.")
    parser.add_argument("--aws", action="store_true",
                        help="sync from the AWS Open Data Registry")
    args = parser.parse_args()

    DEST.mkdir(parents=True, exist_ok=True)
    present = sorted(DEST.glob("*.csv"))
    if present:
        print(f"{len(present)} CSV file(s) already in {DEST}:")
        for p in present:
            print(f"  {p.name:60s} {p.stat().st_size / 1_048_576:8.1f} MB")
        print("\nNext: python -m src.preprocessing.cse_cic_ids2018")
        return 0

    if args.aws:
        return sync_from_aws()

    print(INSTRUCTIONS)
    return 1


if __name__ == "__main__":
    sys.exit(main())
