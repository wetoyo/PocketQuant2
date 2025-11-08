# scraper/utils/filter.py
from pathlib import Path
from typing import List, Optional, Union
import sys
import pandas as pd
root = Path(__file__).parent.parent.parent  # project_root
sys.path.append(str(root))
from configs.paths import DATA_RAW, DATA_PROCESSED

def filter_data(
    file_path: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    include_columns: Optional[List[str]] = None,
    drop_columns: Optional[List[str]] = None,
    file_format: Optional[str] = None
):
    # Auto-detect file path
    if file_path is None:
        raise ValueError("file_path must be provided if not using default folder")
    file_path = Path(file_path)
    
    if file_format is None:
        file_format = file_path.suffix[1:].lower()  # remove the dot
    
    # Read CSV or Parquet
    if file_format == "csv":
        df = pd.read_csv(file_path, parse_dates=True)
    elif file_format in ["parquet", "parq"]:
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")

    # Filter by date range
    if start_date or end_date:
        if "DATE" not in df.columns:
            raise ValueError("No 'DATE' column found for time filtering")
        df["DATE"] = pd.to_datetime(df["DATE"])
        if start_date:
            df = df[df["DATE"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["DATE"] <= pd.to_datetime(end_date)]

    # Include / drop columns
    if include_columns:
        df = df[[col for col in include_columns if col in df.columns]]
    if drop_columns:
        df = df.drop(columns=[col for col in drop_columns if col in df.columns])

    # Default output path
    if output_path is None:
        output_path = DATA_PROCESSED / file_path.name
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save filtered data
    if file_format == "csv":
        df.to_csv(output_path, index=False)
    else:
        df.to_parquet(output_path, index=False)

    return df

# Example usage
if __name__ == "__main__":
    # Filter a specific file for AAPL, only keep DATE and CLOSE, for 2023
    df_filtered = filter_data(
        file_path= DATA_RAW / "AAPL.csv",
        start_date="2023-01-01",
        end_date="2023-12-31",
        include_columns=["DATE", "CLOSE"]
    )
    print(df_filtered.head())