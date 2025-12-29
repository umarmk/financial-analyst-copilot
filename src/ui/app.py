import streamlit as st
import pandas as pd
import sys
import os

from dotenv import load_dotenv

# Get path to project root 
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.llm.client import LLMRequest
from src.llm.factory import get_llm_client
from src.llm.prompts import build_metrics_prompt

load_dotenv()

from src.metrics.core import (
    load_customer_month_mrr,
    load_revenue_events,
    get_mrr_by_month,
    get_mrr_components_by_month,
    get_net_new_mrr,
    get_revenue_churn_rate,
    get_active_customers,
)

# Compute all monthly metrics in one table
def compute_metrics() -> pd.DataFrame:
    customer_month_mrr_df = load_customer_month_mrr()
    events_df = load_revenue_events()

    mrr_by_month = get_mrr_by_month(customer_month_mrr_df)
    components = get_mrr_components_by_month(events_df)
    net_new = get_net_new_mrr(components)
    active = get_active_customers(customer_month_mrr_df)
    churn_rate = get_revenue_churn_rate(components, mrr_by_month)

    metrics_df = (
        mrr_by_month
        .merge(components, on="month", how="left")
        .merge(net_new, on="month", how="left")
        .merge(active, on="month", how="left")
        .merge(churn_rate, on="month", how="left")
    )

    # Convert month label to datetime for plotting
    metrics_df["month_date"] = pd.to_datetime(metrics_df["month"] + "-01")
    metrics_df = metrics_df.sort_values("month_date")

    return metrics_df

# Run the Streamlit metrics explorer
def main() -> None:
    st.set_page_config(
        page_title="Financial Analyst Copilot - Metrics Explorer",
        layout="wide",
    )

    st.title("SaaS Revenue Metrics Explorer")

    # Cache metrics computation
    @st.cache_data
    def get_metrics_df() -> pd.DataFrame:
        return compute_metrics()
    
    with st.spinner("Loading metrics..."):
        metrics_df = get_metrics_df()

    metric_options = {
        "Total MRR": "mrr_total",
        "New MRR": "new_mrr",
        "Expansion MRR": "expansion_mrr",
        "Contraction MRR": "contraction_mrr",
        "Churn MRR": "churn_mrr",
        "Net New MRR": "net_new_mrr",
        "Active Customers": "active_customers",
        "Revenue Churn Rate": "revenue_churn_rate",
    }

    st.sidebar.header("Controls")
    selected_label = st.sidebar.selectbox(
        "Metric",
        list(metric_options.keys()),
    )
    selected_column = metric_options[selected_label]

    # Select time window
    window_choice = st.sidebar.selectbox(
        "Time window",
        options=["Last 6 months", "Last 12 months", "Last 24 months", "All"],
        index=1,
    )

    window_map = {
        "Last 6 months": 6,
        "Last 12 months": 12,
        "Last 24 months": 24,
        "All": None,
    }
    n_months = window_map[window_choice]

    plot_df = metrics_df.copy() 
    if n_months is not None:    # Select time window
        plot_df = plot_df.tail(n_months)


    st.subheader(selected_label)

    # Prepare data for chart
    chart_df = plot_df[["month_date", selected_column]].copy()
    chart_df = chart_df.set_index("month_date")

    st.line_chart(chart_df)

    st.markdown("### Monthly values")
    display_df = plot_df[["month", selected_column]].copy()
    st.dataframe(display_df, use_container_width=True)

    st.markdown("---")
    st.subheader("AI Explanation")

    user_question = st.text_input("Ask a question about the selected metric (optional)", value="")

    explain_clicked = st.button("Explain", type="primary")

    if explain_clicked:
        try:
            client = get_llm_client()

            # Use the same window as the chart
            window_df = metrics_df.copy()
            if n_months is not None:
                window_df = window_df.tail(n_months)

            prompt = build_metrics_prompt(
                window_df=window_df,
                metric_label=selected_label,
                metric_col=selected_column,
                user_question=user_question,
            )

            response = client.generate(
                LLMRequest(
                    prompt=prompt,
                    temperature=0.2,
                    max_tokens=400,
                )
            )

            st.markdown(response if response else "No response returned by the model.")

            with st.expander("Show prompt (debug)"):
                st.code(prompt)

        except Exception as e:
            st.error(f"LLM error: {e}")


if __name__ == "__main__":
    main()


