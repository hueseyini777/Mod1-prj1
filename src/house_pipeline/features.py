"""Feature engineering steps for the King County house sales dataset.

Each function takes a DataFrame and returns a copy with new column(s) added,
so the steps can be composed, reordered, or tested independently. See
``engineer_features`` for the full sequence used by the notebook.
"""

import numpy as np
import pandas as pd

from house_pipeline.config import (
    EARTH_RADIUS_KM,
    WEALTH_CENTER_LAT,
    WEALTH_CENTER_LONG,
)


def add_sqft_price(df: pd.DataFrame) -> pd.DataFrame:
    """Add `sqft_price`: price per square foot of living space and lot."""
    df = df.copy()
    df["sqft_price"] = (df["price"] / (df["sqft_living"] + df["sqft_lot"])).round(2)
    return df


def add_center_distance(
    df: pd.DataFrame,
    ref_lat: float = WEALTH_CENTER_LAT,
    ref_long: float = WEALTH_CENTER_LONG,
) -> pd.DataFrame:
    """Add the distance (km) from each house to a fixed reference point.

    Used as a rough proxy for distance to the "center of wealth" (Bill
    Gates's residence in Medina). Longitude is corrected by `cos(lat)`
    because a degree of longitude covers less physical distance away from
    the equator.
    """
    df = df.copy()
    df["delta_lat"] = np.absolute(ref_lat - df["lat"])
    df["delta_long"] = np.absolute(ref_long - df["long"])
    df["center_distance"] = (
        (df["delta_long"] * np.cos(np.radians(ref_lat))) ** 2 + df["delta_lat"] ** 2
    ) ** 0.5 * 2 * np.pi * EARTH_RADIUS_KM / 360
    return df


def add_water_distance(df: pd.DataFrame) -> pd.DataFrame:
    """Add `water_distance`: distance (km) to the nearest waterfront house.

    For each house, the distance to every waterfront house is computed and
    the minimum is kept. This is only an approximation of proximity to
    water, since waterfront houses do not trace the full shoreline.
    """
    df = df.copy()
    waterfront = df.loc[df["waterfront"] == 1]

    long = df["long"].to_numpy()[:, None]
    lat = df["lat"].to_numpy()[:, None]
    ref_long = waterfront["long"].to_numpy()[None, :]
    ref_lat = waterfront["lat"].to_numpy()[None, :]

    delta_long_corr = (long - ref_long) * np.cos(np.radians(ref_lat))
    delta_lat = lat - ref_lat
    distances = (delta_long_corr**2 + delta_lat**2) ** 0.5 * 2 * np.pi * EARTH_RADIUS_KM / 360

    df["water_distance"] = distances.min(axis=1)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full feature engineering sequence used in the notebook, in order."""
    df = add_sqft_price(df)
    df = add_center_distance(df)
    df = add_water_distance(df)
    return df
