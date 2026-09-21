"""Data cleaning steps for the King County house sales dataset.

Each function takes a DataFrame and returns a cleaned copy, so the steps can
be composed, reordered, or tested independently. See ``clean_data`` for the
full sequence used by the notebook.
"""

import pandas as pd


def drop_bedroom_outliers(df: pd.DataFrame, max_bedrooms: int = 30) -> pd.DataFrame:
    """Drop rows with an implausible bedroom count (e.g. the 33-bedroom record)."""
    return df.loc[df["bedrooms"] <= max_bedrooms].copy()


def fix_sqft_basement(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute `sqft_basement` as `sqft_living - sqft_above`.

    The raw column is stored as text because some rows contain "?" instead
    of a number, so it is rebuilt from two reliable numeric columns instead.
    """
    df = df.copy()
    df["sqft_basement"] = df["sqft_living"] - df["sqft_above"]
    return df


def fill_missing_view(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing `view` values with 0, the overwhelmingly dominant category."""
    df = df.copy()
    df["view"] = df["view"].fillna(0)
    return df


def fill_missing_waterfront(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing `waterfront` values with 0, the overwhelmingly dominant category."""
    df = df.copy()
    df["waterfront"] = df["waterfront"].fillna(0)
    return df


def add_last_known_change(df: pd.DataFrame) -> pd.DataFrame:
    """Replace `yr_built`/`yr_renovated` with a single `last_known_change` column.

    `yr_renovated` is 0 or missing for most houses, so it is combined with
    `yr_built` into one feature: the renovation year when known, otherwise
    the year the house was built.
    """
    df = df.copy()
    renovated = pd.to_numeric(df["yr_renovated"], errors="coerce")
    df["last_known_change"] = renovated.where(
        renovated.notna() & (renovated != 0), df["yr_built"]
    ).astype(int)
    return df.drop(columns=["yr_renovated", "yr_built"])


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full cleaning sequence used in the notebook, in order."""
    df = drop_bedroom_outliers(df)
    df = fix_sqft_basement(df)
    df = fill_missing_view(df)
    df = fill_missing_waterfront(df)
    df = add_last_known_change(df)
    return df
