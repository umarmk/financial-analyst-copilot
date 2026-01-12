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
from src.llm.factory import get_llm_client, get_llm_provider
from src.llm.prompts import build_metrics_prompt, build_executive_summary_prompt

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

# Format currency values
def format_currency(value: float) -> str:
    abs_val = abs(value)
    if abs_val >= 1_000_000_000:
        return f"${value/1_000_000_000:.2f}B"
    if abs_val >= 1_000_000:
        return f"${value/1_000_000:.2f}M"
    if abs_val >= 1_000:
        return f"${value/1_000:.2f}K"
    return f"${value:,.0f}"

# Format metric value
def format_metric_value(metric_col: str, value: float) -> str:
    if metric_col == "revenue_churn_rate":
        return format_percent(float(value))
    return format_currency(float(value))

# Value is in decimal form (e.g., 0.0123 -> 1.23%)
def format_percent(value: float) -> str:
    return f"{value*100:.2f}%"

# Cleanup LLM output to remove common formatting issues
def cleanup_llm_output(text: str, provider: str) -> str:
    """
    Fix common formatting issues from local models.
    """
    cleaned = text.strip()

    if provider == "ollama":
        # Remove triple backtick code fences if the model accidentally uses them
        cleaned = cleaned.replace("```", "")

        # Some models repeat "Top 3 takeaways" twice; keep the first occurrence
        marker = "Top 3 takeaways"
        first = cleaned.find(marker)
        if first != -1:
            second = cleaned.find(marker, first + 1)
            if second != -1:
                cleaned = cleaned[:second].strip()

    return cleaned

# Sanitize Markdown text to prevent Streamlit from treating underscores as italics
def sanitize_markdown(text: str) -> str:
    """
    Escape underscores only outside inline code blocks marked by backticks.
    This prevents accidental italics while keeping `mrr_total` clean.
    """
    result = []
    in_code = False

    for ch in text:
        if ch == "`":
            in_code = not in_code
            result.append(ch)
            continue

        if (not in_code) and ch == "_":
            result.append("\\_")
        else:
            result.append(ch)

    return "".join(result)

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

    # Summary stats for the selected metric in the chosen window
    metric_series = plot_df[selected_column].astype(float)

    current_val = float(metric_series.iloc[-1])
    prev_val = float(metric_series.iloc[-2]) if len(metric_series) >= 2 else current_val
    mom_change = current_val - prev_val
    mom_pct = (mom_change / prev_val) if prev_val != 0 else 0.0

    best_idx = metric_series.idxmax()
    worst_idx = metric_series.idxmin()

    best_month = str(plot_df.loc[best_idx, "month"])
    worst_month = str(plot_df.loc[worst_idx, "month"])

    is_rate = selected_column == "revenue_churn_rate"

    # Format value
    def fmt(v: float) -> str:
        return format_percent(v) if is_rate else format_currency(v)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current", fmt(current_val))
    c2.metric("MoM change", fmt(mom_change), f"{mom_pct*100:.2f}%")
    c3.metric("Best month", best_month)
    c4.metric("Worst month", worst_month)

    # Prepare data for chart
    chart_df = plot_df[["month_date", selected_column]].copy()
    chart_df = chart_df.set_index("month_date")

    st.line_chart(chart_df)

    st.markdown("### Monthly values")
    display_df = plot_df[["month", selected_column]].copy()
    display_df = display_df.rename(columns={selected_column: selected_label})

    # Format values as strings for clean display
    display_df[selected_label] = display_df[selected_label].astype(float).apply(
        lambda v: format_metric_value(selected_column, v)
    )

    st.dataframe(display_df, use_container_width=True)

    st.markdown("---")
    st.subheader("AI Explanation")

    user_question = st.text_input("Ask a question about the selected metric (optional)", value="")
    show_prompt = st.checkbox("Show prompt (debug)", value=False)

    with st.container(horizontal=True):
        explain_clicked = st.button("Explain", type="primary")
        summary_clicked = st.button("Executive Summary")

    # Run LLM
    if explain_clicked or summary_clicked:
        try:
            with st.spinner("Thinking..."):
                client = get_llm_client()

                window_df = metrics_df.copy()
                if n_months is not None:
                    window_df = window_df.tail(n_months)

                if summary_clicked:
                    prompt = build_executive_summary_prompt(
                        window_df=window_df,
                        user_question=user_question,
                    )
                else:
                    prompt = build_metrics_prompt(
                        window_df=window_df,
                        metric_label=selected_label,
                        metric_col=selected_column,
                        user_question=user_question,
                    )

                # Configure LLM
                provider = get_llm_provider()
                temperature = 0.2
                max_tokens = 1200

                if provider == "ollama":
                    temperature = 0.1
                    max_tokens = 1000

                # Generate response
                response = client.generate(
                    LLMRequest(
                        prompt=prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                )

                # Cleanup and display response
                response = cleanup_llm_output(response, provider)
                response = sanitize_markdown(response)
                st.markdown(response if response else "No response returned by the model.")

            if show_prompt:
                st.code(prompt)

        except Exception as e:
            st.error(f"LLM error: {e}")


if __name__ == "__main__":
    main()


