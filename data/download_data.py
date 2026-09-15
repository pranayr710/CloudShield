"""Download UNSW-NB15 dataset into data/raw/.

Run: python data/download_data.py

This script attempts to download from Kaggle. If the Kaggle API is not configured,
it provides manual download instructions.
"""
import os
import sys
from pathlib import Path

# Expected files and their properties
EXPECTED_FILES = {
    "UNSW_NB15_training-set.csv": {"rows": 175341, "cols": 45},
    "UNSW_NB15_testing-set.csv": {"rows": 82332, "cols": 45},
}

DATA_DIR = Path(__file__).parent / "raw"


def check_kaggle_api():
    """Check if Kaggle API is available and configured."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        return True
    except Exception:
        return False


def download_from_kaggle():
    """Download dataset from Kaggle."""
    from kaggle.api.kaggle_api_extended import KaggleApi
    
    api = KaggleApi()
    api.authenticate()
    
    print("Downloading UNSW-NB15 from Kaggle...")
    api.dataset_download_files(
        "mrwellsdavid/unsw-nb15",
        path=str(DATA_DIR),
        unzip=True
    )
    print("Download complete!")


def verify_files():
    """Verify downloaded files have expected shapes."""
    import pandas as pd
    
    all_ok = True
    for filename, expected in EXPECTED_FILES.items():
        filepath = DATA_DIR / filename
        if not filepath.exists():
            print(f"MISSING: {filename}")
            all_ok = False
            continue
        
        df = pd.read_csv(filepath)
        actual_rows, actual_cols = df.shape
        
        if actual_rows != expected["rows"] or actual_cols != expected["cols"]:
            print(f"MISMATCH {filename}: expected {expected['rows']}x{expected['cols']}, got {actual_rows}x{actual_cols}")
            all_ok = False
        else:
            print(f"OK: {filename} ({actual_rows}x{actual_cols})")
    
    return all_ok


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if files already exist
    existing = [f.name for f in DATA_DIR.glob("*.csv")]
    if all(f in existing for f in EXPECTED_FILES):
        print("Dataset files already exist. Verifying...")
        if verify_files():
            print("All files verified. Ready to proceed.")
            return
        else:
            print("Verification failed. Please re-download.")
            sys.exit(1)
    
    # Try Kaggle API
    if check_kaggle_api():
        try:
            download_from_kaggle()
            if verify_files():
                print("Dataset downloaded and verified successfully!")
                return
        except Exception as e:
            print(f"Kaggle download failed: {e}")
    
    # Manual fallback instructions
    print("\n" + "="*60)
    print("MANUAL DOWNLOAD REQUIRED")
    print("="*60)
    print(f"""
Please download the UNSW-NB15 dataset manually:

Option 1 - Kaggle (recommended):
  1. Go to: https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15/data
  2. Download the dataset
  3. Extract and place these files in: {DATA_DIR}
     - UNSW_NB15_training-set.csv
     - UNSW_NB15_testing-set.csv

Option 2 - Official UNSW source:
  1. Go to: https://research.unsw.edu.au/projects/unsw-nb15-dataset
  2. Download from the SharePoint link
  3. Place the two CSV files in: {DATA_DIR}

IMPORTANT: Do NOT use UNSW-NB15_1.csv through _4.csv (raw dump, 49 cols, no header).
Use only the two partition files (45 columns, with headers).

Expected file sizes: ~15 MB (training) and ~31 MB (testing)
""")
    sys.exit(1)


if __name__ == "__main__":
    main()
