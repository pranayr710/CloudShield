"""Run the full pipeline end to end:  python -m src.preprocessing"""

from .pipeline import export_processed, get_dataset

if __name__ == "__main__":
    export_processed()
    for task in ("regression", "classification", "clustering"):
        print("\n" + "=" * 60)
        get_dataset(task, verbose=True)
