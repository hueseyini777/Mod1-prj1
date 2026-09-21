"""CLI: run the cleaning + feature engineering pipeline and save the result.

Usage:
    uv run python scripts/build_dataset.py [output_csv_path]
"""

import sys
from pathlib import Path

from house_pipeline.pipeline import build_dataset

DEFAULT_OUTPUT_PATH = Path("data/king_county_processed.csv")


def main() -> None:
    output_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset()
    dataset.to_csv(output_path, index=False)
    print(f"Wrote {len(dataset)} rows x {len(dataset.columns)} columns to {output_path}")


if __name__ == "__main__":
    main()
