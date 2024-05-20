"""
Preprocess raw KSP FIR data into a clean motor vehicle accident dataset.

Usage:
    python3 preprocess.py

Expects:
    data/FIR_Details_Data.csv

Outputs:
    processed_data.csv
"""

import pandas as pd

DATA_DIR = "data"
FIR_FILE = f"{DATA_DIR}/FIR_Details_Data.csv"
OUTPUT_FILE = "processed_data.csv"

ROAD_TYPE_NORM = {
    "INDIAN MOTOR VEHICLE (IMV)": "Other Roads",
}

VALID_ROAD_TYPES = {"National Highways", "State Highways", "Other Roads", "Other Places"}


def derive_severity(crime_group: str) -> str:
    cg = str(crime_group).upper()
    if "NON-FATAL" in cg or "NON FATAL" in cg:
        return "Non-Fatal"
    if "FATAL" in cg:
        return "Fatal"
    return "Non-Fatal"


def main() -> None:
    print(f"Loading {FIR_FILE} ...")
    df = pd.read_csv(
        FIR_FILE,
        usecols=[
            "District_Name", "UnitName", "FIRNo", "Year", "Month",
            "CrimeGroup_Name", "CrimeHead_Name",
        ],
        low_memory=False,
    )
    print(f"  Total rows: {len(df):,}")

    # Filter to motor vehicle accidents only
    accident_mask = df["CrimeGroup_Name"].str.contains("MOTOR VEHICLE", case=False, na=False)
    df = df[accident_mask].copy()
    print(f"  Accident rows: {len(df):,}")

    # Deduplicate: FIR file has one row per ActSection/charge, not per FIR
    df = df.drop_duplicates(subset=["UnitName", "FIRNo", "Year"])
    print(f"  Unique FIRs after deduplication: {len(df):,}")

    # Derive severity from CrimeGroup_Name
    df["Severity"] = df["CrimeGroup_Name"].map(derive_severity)

    # Normalize road type
    df = df.rename(columns={"CrimeHead_Name": "Road_Type"})
    df["Road_Type"] = df["Road_Type"].map(
        lambda x: ROAD_TYPE_NORM.get(str(x).strip(), str(x).strip())
    )

    # Keep relevant columns, drop rows missing key fields
    keep = ["District_Name", "Road_Type", "Month", "Year", "Severity"]
    df = df[keep].dropna(subset=["District_Name", "Road_Type"])
    df = df[df["Road_Type"].isin(VALID_ROAD_TYPES)]
    df["Month"] = df["Month"].astype(int)
    df["Year"] = df["Year"].astype(int)

    print(f"\nFinal dataset: {len(df):,} rows")
    print("\nSeverity distribution:")
    print(df["Severity"].value_counts().to_string())
    print("\nRoad Type distribution:")
    print(df["Road_Type"].value_counts().to_string())
    print(f"\nDistricts: {df['District_Name'].nunique()}")
    print(f"Years: {sorted(df['Year'].unique().tolist())}")

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
