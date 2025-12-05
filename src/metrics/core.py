import pandas as pd

from src.config import DATA_PROCESSED_DIR

# Loaders

# Load processed customers table
def load_customers() -> pd.DataFrame:
    path = DATA_PROCESSED_DIR / "customers.csv"
    return pd.read_csv(path, parse_dates=["signup_date"])

# Load monthly MRR per customer
def load_customer_month_mrr() -> pd.DataFrame:
    path = DATA_PROCESSED_DIR / "customer_month_mrr.csv"
    return pd.read_csv(path)

# Load revenue events
def load_revenue_events() -> pd.DataFrame:
    path = DATA_PROCESSED_DIR / "revenue_events.csv"
    return pd.read_csv(path, parse_dates=["event_date"])

# Core metrics

# Compute total MRR per month
def get_mrr_by_month(customer_month_mrr_df: pd.DataFrame) -> pd.DataFrame:
    df = (
        customer_month_mrr_df.groupby("month", as_index=False)["mrr"]
        .sum()
        .sort_values("month")
    )
    df = df.rename(columns={"mrr": "mrr_total"})
    return df

# Compute New / Expansion / Contraction / Churn MRR per month
def get_mrr_components_by_month(events_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        events_df.groupby(["event_month", "event_type"])["mrr_delta"]
        .sum()
        .unstack(fill_value=0.0)
    )

    for col in ["new", "expansion", "contraction", "churn"]:
        if col not in grouped.columns:
            grouped[col] = 0.0

    grouped = grouped.reset_index().rename(columns={"event_month": "month"})

    result = grouped[["month", "new", "expansion", "contraction", "churn"]].copy()
    result = result.sort_values("month").reset_index(drop=True)

    result = result.rename(
        columns={
            "new": "new_mrr",
            "expansion": "expansion_mrr",
            "contraction": "contraction_mrr",
            "churn": "churn_mrr",
        }
    )

    return result

# Compute Net New MRR per month
def get_net_new_mrr(components_df: pd.DataFrame) -> pd.DataFrame:
    df = components_df.copy()
    df["net_new_mrr"] = (
        df["new_mrr"]
        + df["expansion_mrr"]
        + df["contraction_mrr"]
        + df["churn_mrr"]
    )
    return df[["month", "net_new_mrr"]]

# Compute revenue churn rate per month: |Churn| / MRR(prev_month)
def get_revenue_churn_rate(
    components_df: pd.DataFrame,
    mrr_by_month_df: pd.DataFrame,
) -> pd.DataFrame:
    churn_df = components_df[["month", "churn_mrr"]].copy()
    mrr_df = mrr_by_month_df.copy()

    churn_df = churn_df.sort_values("month").reset_index(drop=True)
    mrr_df = mrr_df.sort_values("month").reset_index(drop=True)

    churn_df["mrr_prev"] = mrr_df["mrr_total"].shift(1)
    churn_df["mrr_prev"] = churn_df["mrr_prev"].fillna(0.0)

    def safe_rate(row: pd.Series) -> float:
        if row["mrr_prev"] <= 0:
            return 0.0
        return abs(row["churn_mrr"]) / row["mrr_prev"]

    churn_df["revenue_churn_rate"] = churn_df.apply(safe_rate, axis=1)

    return churn_df[["month", "revenue_churn_rate"]]

# Count active customers per month
def get_active_customers(customer_month_mrr_df: pd.DataFrame) -> pd.DataFrame:
    df = customer_month_mrr_df.copy()
    df["is_active"] = df["mrr"] > 0

    active_df = (
        df.groupby("month", as_index=False)["is_active"]
        .sum()
        .rename(columns={"is_active": "active_customers"})
        .sort_values("month")
    )

    return active_df

    
