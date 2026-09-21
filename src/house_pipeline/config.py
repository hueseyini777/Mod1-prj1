"""Shared constants for the King County house sales pipeline."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "King_County_House_prices_dataset.csv"
MODEL_DIR = PROJECT_ROOT / "model"
MODEL_PATH = MODEL_DIR / "model.bin"

# Reference point used as a proxy "center of wealth": Bill Gates's residence
# in Medina, on Lake Washington.
WEALTH_CENTER_LAT = 47.62774
WEALTH_CENTER_LONG = -122.24194

EARTH_RADIUS_KM = 6378

# Columns dropped before modeling because they leak the target or add no
# predictive value.
LEAKAGE_COLUMNS = ["price", "sqft_price", "date", "delta_lat", "delta_long"]
