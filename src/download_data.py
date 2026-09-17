"""
Download the Customer Support on Twitter dataset via kagglehub.
Falls back to reading from data/raw/ if already downloaded.
"""
import shutil
from pathlib import Path
from src.utils import get_paths


def download():
    paths = get_paths()
    target = paths["raw"] / "twcs.csv"

    if target.exists():
        print(f"Dataset already present at {target}")
        return target

    print("Downloading dataset from Kaggle...")
    import kagglehub

    cache_path = Path(kagglehub.dataset_download("thoughtvector/customer-support-on-twitter"))
    print(f"Downloaded to cache: {cache_path}")

    # kagglehub returns a folder; find twcs.csv
    csv_file = None
    for p in cache_path.rglob("*.csv"):
        if p.name.lower() == "twcs.csv":
            csv_file = p
            break

    if csv_file is None:
        # sometimes the file is named differently
        csvs = list(cache_path.rglob("*.csv"))
        if not csvs:
            raise FileNotFoundError("No CSV found in Kaggle download")
        csv_file = csvs[0]

    print(f"Copying {csv_file} -> {target}")
    shutil.copy(csv_file, target)
    print("Done.")
    return target


if __name__ == "__main__":
    download()