import numpy as np
import pandas as pd
import pytest

from house_pipeline.cleaning import (
    add_last_known_change,
    clean_data,
    drop_bedroom_outliers,
    fill_missing_view,
    fill_missing_waterfront,
    fix_sqft_basement,
)


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "date": ["20140502T000000", "20140503T000000", "20140504T000000"],
            "price": [300000.0, 450000.0, 600000.0],
            "bedrooms": [3, 33, 4],
            "bathrooms": [2.0, 2.0, 2.5],
            "sqft_living": [1500, 1620, 2200],
            "sqft_lot": [5000, 6000, 7000],
            "floors": [1.0, 1.0, 2.0],
            "waterfront": [0.0, np.nan, 1.0],
            "view": [0.0, np.nan, 2.0],
            "condition": [3, 4, 3],
            "grade": [7, 6, 9],
            "sqft_above": [1500, 1620, 1800],
            "sqft_basement": ["0", "?", "400"],
            "yr_built": [1990, 1955, 2005],
            "yr_renovated": [0.0, np.nan, 2010.0],
            "zipcode": [98103, 98002, 98004],
            "lat": [47.65, 47.30, 47.62],
            "long": [-122.35, -122.20, -122.24],
            "sqft_living15": [1500, 1600, 2100],
            "sqft_lot15": [5000, 6000, 7000],
        }
    )


def test_drop_bedroom_outliers_removes_implausible_rows(raw_df):
    cleaned = drop_bedroom_outliers(raw_df, max_bedrooms=30)

    assert cleaned["bedrooms"].max() <= 30
    assert 33 not in cleaned["bedrooms"].values
    assert len(cleaned) == len(raw_df) - 1


def test_fix_sqft_basement_recomputes_from_living_and_above(raw_df):
    cleaned = fix_sqft_basement(raw_df)

    expected = raw_df["sqft_living"] - raw_df["sqft_above"]
    pd.testing.assert_series_equal(
        cleaned["sqft_basement"], expected, check_names=False
    )
    assert pd.api.types.is_numeric_dtype(cleaned["sqft_basement"])


def test_fill_missing_view_and_waterfront_use_zero(raw_df):
    cleaned = fill_missing_waterfront(fill_missing_view(raw_df))

    assert cleaned["view"].isna().sum() == 0
    assert cleaned["waterfront"].isna().sum() == 0
    assert cleaned.loc[1, "view"] == 0
    assert cleaned.loc[1, "waterfront"] == 0


def test_add_last_known_change_prefers_renovation_year(raw_df):
    cleaned = add_last_known_change(raw_df)

    assert "yr_built" not in cleaned.columns
    assert "yr_renovated" not in cleaned.columns
    # Row 0: yr_renovated == 0 -> falls back to yr_built.
    assert cleaned.loc[0, "last_known_change"] == 1990
    # Row 1: yr_renovated missing -> falls back to yr_built.
    assert cleaned.loc[1, "last_known_change"] == 1955
    # Row 2: yr_renovated present and nonzero -> used as-is.
    assert cleaned.loc[2, "last_known_change"] == 2010


def test_clean_data_runs_full_sequence_without_missing_values(raw_df):
    cleaned = clean_data(raw_df)

    assert cleaned["view"].isna().sum() == 0
    assert cleaned["waterfront"].isna().sum() == 0
    assert cleaned["bedrooms"].max() <= 30
    assert "yr_built" not in cleaned.columns
    assert "yr_renovated" not in cleaned.columns
    assert "last_known_change" in cleaned.columns
    assert pd.api.types.is_numeric_dtype(cleaned["sqft_basement"])
