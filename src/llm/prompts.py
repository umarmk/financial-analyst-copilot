from __future__ import annotations

import pandas as pd 

# Helper function
def _to_markdown_table(df: pd.DataFrame, max_rows: int = 24) -> str:
    trimmed = df.tail(max_rows).copy()
    return trimmed.to_markdown(index=False)

# Single-metric business performance summary
def build_metrics_prompt(
    window_df: pd.DataFrame,
    metric_label: str,
    metric_col: str,
    user_question: str | None,
) -> str:
    """
    Build a constrained prompt:
    - Explain only what is visible in the data.
    - Do not invent reasons (marketing, pricing, etc.).
    - If asked "why", say the data doesn't include causes.
    """
    df = window_df.copy()

    # Basic facts (helps reduce hallucinations)
    start_month = str(df["month"].iloc[0])
    end_month = str(df["month"].iloc[-1])

    start_val = float(df[metric_col].iloc[0])
    end_val = float(df[metric_col].iloc[-1])
    delta = end_val - start_val

    mom = df[metric_col].diff()
    # Biggest month-over-month change (absolute)
    idx = mom.abs().idxmax()
    biggest_change_month = str(df.loc[idx, "month"]) if pd.notna(idx) else "N/A"
    biggest_change_val = float(mom.loc[idx]) if pd.notna(idx) else 0.0

    # Small table: always include Total MRR and Net New MRR for context if present
    cols = ["month", metric_col]
    for extra in ["mrr_total", "net_new_mrr", "new_mrr", "expansion_mrr", "contraction_mrr", "churn_mrr"]:
        if extra in df.columns and extra not in cols:
            cols.append(extra)

    table_df = df[cols].copy()
    table_md = _to_markdown_table(table_df, max_rows=24)

    question_block = ""
    if user_question and user_question.strip():
        question_block = f"\nUser question: {user_question.strip()}\n"

    prompt = f"""
You are a SaaS finance analyst. Use ONLY the provided data.
Do not invent causes or assumptions (no marketing, pricing, product changes, etc.).
If asked "why", explain that the dataset does not contain causal drivers.

Metric to explain: {metric_label} ({metric_col})
Time window: {start_month} to {end_month}

Facts (computed from the data):
- Start value: {start_val}
- End value: {end_val}
- Change over window: {delta}
- Biggest MoM change: {biggest_change_val} in {biggest_change_month}

Data table:
{table_md}
{question_block}

Write 5-10 bullet points:
- Trend summary
- Notable spikes/drops (month + magnitude)
- Relationship to Net New MRR / Churn MRR when relevant
- Keep it plain language for a finance stakeholder

Formatting rules:
- Use Markdown.
- Use '-' for bullets.
- Do NOT use underscores for emphasis.
- If you mention column names, wrap them in backticks (example: `net_new_mrr`).
- Keep spaces between numbers and units (example: 1.54 M).
""".strip()

    return prompt

# Multi-metric business performance summary
def build_executive_summary_prompt(
    window_df: pd.DataFrame,
    user_question: str | None,
) -> str:
    """
    Multi-metric business performance summary for the selected time window.
    """
    df = window_df.copy()
    start_month = str(df["month"].iloc[0])
    end_month = str(df["month"].iloc[-1])

    # Keep table compact and consistent
    cols = [
        "month",
        "mrr_total",
        "net_new_mrr",
        "new_mrr",
        "expansion_mrr",
        "contraction_mrr",
        "churn_mrr",
        "active_customers",
        "revenue_churn_rate",
    ]
    cols = [c for c in cols if c in df.columns]
    table_md = _to_markdown_table(df[cols], max_rows=24)

    question_block = ""
    if user_question and user_question.strip():
        question_block = f"\nUser question: {user_question.strip()}\n"

    prompt = f"""
You are a SaaS finance analyst. Use ONLY the provided data.
Do not invent causes or assumptions (no marketing, pricing, product changes, etc.).
If asked "why", explain that the dataset does not contain causal drivers.

Formatting rules (must follow):
- Output ONLY Markdown bullet points using '- '.
- Do NOT use headings (no '#', '##', etc.).
- Do NOT use numbered lists.
- Do NOT use code blocks (no triple backticks).
- Do NOT repeat the same section twice.
- If you mention column names, wrap them in backticks (example: `net_new_mrr`).

Task: Write an executive summary of business performance from {start_month} to {end_month}.
Focus on:
- Overall MRR trend (start, end, change)
- What drove Net New MRR (New vs Expansion vs Churn vs Contraction)
- Any notable spikes/drops (month + magnitude)
- Active customers trend and revenue churn rate trend (if meaningful)

Data table:
{table_md}
{question_block}

Output:
- 8–12 bullet points
- Then a short “Top 3 takeaways” section
""".strip()

    return prompt