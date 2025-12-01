from pathlib import Path
import pandas as pd

from src.config import DATA_RAW_DIR, DATA_PROCESSED_DIR

# Transform accounts into canonical customers table
def build_customers(accounts_df: pd.DataFrame) -> pd.DataFrame:

    # Rename core columns to customer schema
    customers_df = accounts_df.rename(
        columns={
            "account_id": "customer_id",
            "account_name": "customer_name",
            "plan_tier": "initial_plan",
        }
    ).copy()

    # Keep only needed columns
    customers_df = customers_df[
        [
            "customer_id",
            "customer_name",
            "industry",
            "country",
            "signup_date",
            "initial_plan",
        ]
    ]

    # Placeholder for is_active (will be updated after MRR is built)
    customers_df["is_active"] = 0

    return customers_df

# Load accounts.csv, build customers.csv, and save it
def main():
    # Paths
    accounts_path = DATA_RAW_DIR / "accounts.csv"
    customers_path = DATA_PROCESSED_DIR / "customers.csv"

    # Load raw accounts data
    accounts_df = pd.read_csv(accounts_path, parse_dates=["signup_date"])

    # Build canonical customers table
    customers_df = build_customers(accounts_df)

    # Ensure processed directory exists
    customers_path.parent.mkdir(parents=True, exist_ok=True)

    # Save customers.csv
    customers_df.to_csv(customers_path, index=False)

    print(f"Saved {len(customers_df)} customers to {customers_path}")

if __name__ == "__main__":
    main()
