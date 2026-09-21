import numpy as np
import pandas as pd
import pytest

from house_pipeline.config import EARTH_RADIUS_KM, WEALTH_CENTER_LAT, WEALTH_CENTER_LONG
from house_pipeline.features import (
    add_center_distance,
    add_sqft_price,
    add_water_distance,
    engineer_features,
)


@pytest.fixture
def cleaned_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "price": [300000.0, 450000.0, 600000.0],
            "sqft_living": [1500, 1620, 2200],
            "sqft_lot": [5000, 6000, 7000],
            "waterfront": [1.0, 0.0, 1.0],
            "lat": [WEALTH_CENTER_LAT, 47.30, 47.62],
            "long": [WEALTH_CENTER_LONG, -122.20, -122.24],
        }
    )


def test_add_sqft_price_is_price_per_living_plus_lot_area(cleaned_df):
    result = add_sqft_price(cleaned_df)

    expected = (cleaned_df["price"] / (cleaned_df["sqft_living"] + cleaned_df["sqft_lot"])).round(2)
    pd.testing.assert_series_equal(result["sqft_price"], expected, check_names=False)


def test_add_center_distance_is_zero_at_the_reference_point(cleaned_df):
    result = add_center_distance(cleaned_df)

    assert result.loc[0, "center_distance"] == pytest.approx(0.0, abs=1e-6)
    assert result.loc[1, "center_distance"] > 0


def test_add_water_distance_is_zero_for_a_waterfront_house_itself(cleaned_df):
    result = add_water_distance(cleaned_df)

    # Rows 0 and 2 are themselves waterfront houses, so their nearest
    # waterfront neighbor (including themselves) is 0 km away.
    assert result.loc[0, "water_distance"] == pytest.approx(0.0, abs=1e-6)
    assert result.loc[2, "water_distance"] == pytest.approx(0.0, abs=1e-6)

    # Row 1 is not on the water; its distance should match the closer of
    # the two waterfront reference points, computed independently.
    def manual_distance(lat, long, ref_lat, ref_long):
        delta_long_corr = (long - ref_long) * np.cos(np.radians(ref_lat))
        delta_lat = lat - ref_lat
        return (delta_long_corr**2 + delta_lat**2) ** 0.5 * 2 * np.pi * EARTH_RADIUS_KM / 360

    d_to_row0 = manual_distance(47.30, -122.20, WEALTH_CENTER_LAT, WEALTH_CENTER_LONG)
    d_to_row2 = manual_distance(47.30, -122.20, 47.62, -122.24)
    assert result.loc[1, "water_distance"] == pytest.approx(min(d_to_row0, d_to_row2))


def test_engineer_features_adds_all_expected_columns(cleaned_df):
    result = engineer_features(cleaned_df)

    for column in ["sqft_price", "center_distance", "water_distance"]:
        assert column in result.columns
        assert result[column].isna().sum() == 0
