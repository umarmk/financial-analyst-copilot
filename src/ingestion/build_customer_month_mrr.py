import pandas as pd

from src.config import DATA_RAW_DIR, DATA_PROCESSED_DIR


# Build monthly MRR per customer from subscriptions
def build_customer_month_mrr(subscriptions_df: pd.DataFrame) -> pd.DataFrame:
    
    # Ensure dates are datetime objects
    subscriptions_df = subscriptions_df.copy()
    subscriptions_df["start_date"] = pd.to_datetime(subscriptions_df["start_date"])
    subscriptions_df["end_date"] = pd.to_datetime(subscriptions_df["end_date"])
    
    # Compute dataset horizon for active subscriptions
    max_start = subscriptions_df["start_date"].max()
    max_end = subscriptions_df["end_date"].max()  # ignores NaT by default

    if pd.isna(max_end):
        global_last_date = max_start
    else:
        global_last_date = max(max_start, max_end)

    records = []

    # Expand each subscription into active months
    for _, row in subscriptions_df.iterrows():
        account_id = row["account_id"]
        start_date = row["start_date"]
        end_date = row["end_date"]
        mrr_amount = row["mrr_amount"]

        # Set effective end date (exclusive for churn month)
        if pd.isna(end_date):
            effective_end = global_last_date
        else:
            effective_end = end_date - pd.Timedelta(days=1)

        # Skip if end before start
        if effective_end < start_date:
            continue

        # Align to month starts
        start_month = start_date.replace(day=1)
        end_month = effective_end.replace(day=1)

        # Generate one row per active month
        month_range = pd.date_range(start=start_month, end=end_month, freq="MS")

        for month_start in month_range:
            month_str = month_start.strftime("%Y-%m")
            records.append(
                {
                    "customer_id": account_id,
                    "month": month_str,
                    "mrr": float(mrr_amount),
                }
            )
        
    # Aggregate MRR per customer per month
    month_mrr_df = (
        pd.DataFrame(records)
        .groupby(["customer_id", "month"], as_index=False)["mrr"]
        .sum()
    )

    return month_mrr_df

# Load subscriptions, build monthly MRR, and save to CSV
def main() -> None:
    subscriptions_path = DATA_RAW_DIR / "subscriptions.csv"
    output_path = DATA_PROCESSED_DIR / "customer_month_mrr.csv"

    # Load raw subscriptions
    subscriptions_df = pd.read_csv(
        subscriptions_path,
        parse_dates=["start_date", "end_date"],
    )

    # Build monthly MRR
    month_mrr_df = build_customer_month_mrr(subscriptions_df)

    # Ensure processed directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    month_mrr_df.to_csv(output_path, index=False)

    print(f"Saved {len(month_mrr_df)} rows to {output_path}")
    print("Month range:", month_mrr_df["month"].min(), "→", month_mrr_df["month"].max())
    print("Customers covered:", month_mrr_df["customer_id"].nunique())

if __name__ == "__main__":
    main()

