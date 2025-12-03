import pandas as pd

from src.config import DATA_PROCESSED_DIR

# Set is_active based on MRR in the latest month

def update_is_active(
    customers_df: pd.DataFrame,
    month_mrr_df: pd.DataFrame,
) -> pd.DataFrame:

    customers = customers_df.copy()
    month_mrr = month_mrr_df.copy()

    # Find latest month label available
    latest_month = month_mrr["month"].max()

    # Filter to latest month
    latest_mrr = month_mrr[month_mrr["month"] == latest_month][
        ["customer_id", "mrr"]
    ]

    # Ensure one row per customer
    latest_mrr = latest_mrr.groupby("customer_id", as_index=False)["mrr"].sum()

    # Merge with customers
    customers = customers.merge(
        latest_mrr,
        on="customer_id",
        how="left",
    )

    # Replace missing MRR with zero
    customers["mrr"] = customers["mrr"].fillna(0.0)

    # Set is_active flag
    customers["is_active"] = (customers["mrr"] > 0).astype(int)

    # Drop helper column
    customers = customers.drop(columns=["mrr"])

    return customers

# Load customers and monthly MRR, update is_active, save customers
def main() -> None:
    customers_path = DATA_PROCESSED_DIR / "customers.csv"
    month_mrr_path = DATA_PROCESSED_DIR / "customer_month_mrr.csv"

    customers_df = pd.read_csv(customers_path, parse_dates=["signup_date"])
    month_mrr_df = pd.read_csv(month_mrr_path)

    updated_customers = update_is_active(customers_df, month_mrr_df)

    updated_customers.to_csv(customers_path, index=False)

    active_count = updated_customers["is_active"].sum()
    total = len(updated_customers)
    print(f"Updated customers.csv: {active_count}/{total} active as of latest month")

if __name__ == "__main__":
    main()

