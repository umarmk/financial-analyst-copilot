import pandas as pd

from src.config import DATA_PROCESSED_DIR

# Build revenue events from monthly MRR snapshot
def build_revenue_events(month_mrr_df: pd.DataFrame) -> pd.DataFrame:
    
    df = month_mrr_df.copy()

    # Convert month label to first day of month
    df["month_start"] = pd.to_datetime(df["month"] + "-01")

    # Global ordered list of all months
    all_months = sorted(df["month_start"].unique())

    records = []

    # Process each customer separately
    for customer_id, group in df.groupby("customer_id"):
        # Map month_start -> mrr for this customer
        mrr_by_month = group.set_index("month_start")["mrr"]

        prev_mrr = 0.0

        for month_start in all_months:
            # Use 0 if customer has no row for this month
            current_mrr = float(mrr_by_month.get(month_start, 0.0))

            # Skip if no change
            if current_mrr == prev_mrr:
                prev_mrr = current_mrr
                continue

            mrr_delta = current_mrr - prev_mrr

            # Classify event type
            if prev_mrr == 0 and current_mrr > 0:
                event_type = "new"
            elif prev_mrr > 0 and current_mrr == 0:
                event_type = "churn"
            elif prev_mrr > 0 and current_mrr > 0 and mrr_delta > 0:
                event_type = "expansion"
            elif prev_mrr > 0 and current_mrr > 0 and mrr_delta < 0:
                event_type = "contraction"
            else:
                # Unexpected pattern, move on
                prev_mrr = current_mrr
                continue

            event_month = month_start.strftime("%Y-%m")
            event_date = (month_start + pd.offsets.MonthEnd(0)).date()

            records.append(
                {
                    "event_id": f"{customer_id}-{event_month}",
                    "customer_id": customer_id,
                    "event_month": event_month,
                    "event_date": event_date.isoformat(),
                    "event_type": event_type,
                    "mrr_delta": float(mrr_delta),
                    "mrr_after_event": float(current_mrr),
                }
            )

            prev_mrr = current_mrr

    events_df = pd.DataFrame(records)

    return events_df

# Load monthly MRR, build events, and save to CSV
def main() -> None:

    month_mrr_path = DATA_PROCESSED_DIR / "customer_month_mrr.csv"
    events_path = DATA_PROCESSED_DIR / "revenue_events.csv"

    # Load monthly MRR snapshot
    month_mrr_df = pd.read_csv(month_mrr_path)

    # Build events
    events_df = build_revenue_events(month_mrr_df)

    # Ensure processed directory exists
    events_path.parent.mkdir(parents=True, exist_ok=True)

    # Save events
    events_df.to_csv(events_path, index=False)

    print(f"Saved {len(events_df)} events to {events_path}")
    print("Event types:\n", events_df["event_type"].value_counts())

if __name__ == "__main__":
    main()
