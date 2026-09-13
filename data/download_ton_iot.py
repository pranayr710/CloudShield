"""Fetch TON_IoT data into data/raw/TON_IoT/.

    python data/download_ton_iot.py

CloudShield Track 3 (clustering, Member 3).

TON_IoT is heterogeneous - IoT/IIoT telemetry, Windows and Linux host logs, and
network traffic. Do NOT concatenate every modality. Member 3 selects and
documents ONE defensible subset for the clustering experiment (spec step 36).
"""

import sys
from pathlib import Path

DEST = Path(__file__).resolve().parent / "raw" / "TON_IoT"

INSTRUCTIONS = f"""
{'=' * 72}
TON_IoT - DOWNLOAD REQUIRED  (CloudShield Track 3, clustering)
{'=' * 72}

Destination: {DEST}

Official page
    https://research.unsw.edu.au/projects/toniot-datasets

Available modalities - pick ONE and document the choice:
    Train_Test_datasets/Train_Test_Network/       network flow records
    Train_Test_datasets/Train_Test_IoT_*/         per-sensor telemetry
    Train_Test_datasets/Train_Test_Windows_*/     Windows host telemetry
    Train_Test_datasets/Train_Test_Linux_*/       Linux host telemetry

Suggested starting point for CloudShield: the processed network subset, as it
is the modality most comparable to the other two tracks. Confirm by inspecting
the files before committing to it.

Attack types include backdoor, ddos, dos, injection, mitm, password,
ransomware, scanning and xss. Confirm the actual label column and its values by
inspection - do not hard-code them.

CLUSTERING RULE (spec section 8)
    Attack labels must NOT be used as clustering inputs. They may be used only
    afterwards, to interpret or validate the discovered clusters.

CITATION
    Moustafa, N. (2021). A new distributed architecture for evaluating AI-based
    security systems at the edge: Network TON_IoT datasets. Sustainable Cities
    and Society, 72, 102994.
{'=' * 72}
"""


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    present = sorted(DEST.rglob("*.csv"))
    if present:
        print(f"{len(present)} CSV file(s) found in {DEST}:")
        for p in present[:20]:
            print(f"  {str(p.relative_to(DEST)):58s} "
                  f"{p.stat().st_size / 1_048_576:8.1f} MB")
        if len(present) > 20:
            print(f"  ... and {len(present) - 20} more")
        print("\nMember 3: select and document ONE modality before clustering.")
        return 0
    print(INSTRUCTIONS)
    return 1


if __name__ == "__main__":
    sys.exit(main())
