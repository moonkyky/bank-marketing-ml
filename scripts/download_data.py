from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

DATA_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"

OUTER_ZIP = RAW_DIR / "bank-marketing.zip"
INNER_ZIP = RAW_DIR / "bank-additional.zip"

EXPECTED_CSV = RAW_DIR / "bank-additional" / "bank-additional-full.csv"


def download_data() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if not OUTER_ZIP.exists():
        print("Downloading UCI Bank Marketing dataset...")
        urlretrieve(DATA_URL, OUTER_ZIP)
        print(f"Saved: {OUTER_ZIP}")

    if not INNER_ZIP.exists():
        print("Extracting outer archive...")

        with ZipFile(OUTER_ZIP) as archive:
            archive.extractall(RAW_DIR)

    if not EXPECTED_CSV.exists():
        print("Extracting bank-additional archive...")

        with ZipFile(INNER_ZIP) as archive:
            archive.extractall(RAW_DIR)

    if not EXPECTED_CSV.exists():
        raise FileNotFoundError(
            f"Expected dataset not found: {EXPECTED_CSV}"
        )

    print(f"Dataset ready: {EXPECTED_CSV}")
    return EXPECTED_CSV


if __name__ == "__main__":
    download_data()
