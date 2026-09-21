"""End-to-end data preparation pipeline for the King County house sales dataset."""

from pathlib import Path

import pandas as pd

from house_pipeline.cleaning import clean_data
from house_pipeline.config import RAW_DATA_PATH
from house_pipeline.features import engineer_features


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw King County house sales CSV."""
    return pd.read_csv(path)


def build_dataset(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw data and apply cleaning and feature engineering, in order."""
    raw = load_raw_data(path)
    cleaned = clean_data(raw)
    return engineer_features(cleaned)
