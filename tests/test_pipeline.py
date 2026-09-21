from house_pipeline.pipeline import build_dataset, load_raw_data


def test_load_raw_data_reads_the_expected_columns():
    raw = load_raw_data()

    assert "sqft_basement" in raw.columns
    assert raw.shape[0] > 0


def test_build_dataset_produces_a_clean_and_engineered_frame():
    dataset = build_dataset()

    # Cleaning: no more implausible bedroom counts, no missing waterfront/view,
    # yr_built/yr_renovated replaced by last_known_change.
    assert dataset["bedrooms"].max() <= 30
    assert dataset["waterfront"].isna().sum() == 0
    assert dataset["view"].isna().sum() == 0
    assert "yr_built" not in dataset.columns
    assert "yr_renovated" not in dataset.columns
    assert "last_known_change" in dataset.columns

    # sqft_basement was reconstructed and is fully numeric.
    assert dataset["sqft_basement"].dtype.kind in "if"

    # Feature engineering: new columns present with no missing values.
    for column in ["sqft_price", "center_distance", "water_distance"]:
        assert column in dataset.columns
        assert dataset[column].isna().sum() == 0
        assert (dataset[column] >= 0).all()
