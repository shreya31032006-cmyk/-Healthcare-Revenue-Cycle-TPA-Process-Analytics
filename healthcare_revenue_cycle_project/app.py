"""
Healthcare Revenue Cycle and TPA Process Analytics Dashboard
=============================================================
A Python-only Streamlit dashboard for monitoring revenue-cycle activities,
claim status, TPA performance, and process analytics.
Includes a Random Forest-based claim rejection predictor.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ---------------------------------------------------------------------------
# Constants & Project Assumptions
# ---------------------------------------------------------------------------
DATASET_PATH = os.path.join(os.path.dirname(__file__), "data", "claims_data.csv")

REQUIRED_COLUMNS = [
    "Case_ID", "Patient_Type", "Department", "TPA_or_Payer", "Service_Type",
    "Claim_Amount", "Approved_Amount", "Rejected_Amount", "Pending_Amount",
    "Submission_Date", "Settlement_Date", "Case_Status", "Query_or_Rejection_Reason",
]

VALID_STATUSES = {"Approved", "Rejected", "Pending", "Partially Approved"}

# Project Assumption: A case is considered DELAYED if:
#   - Status is Pending AND Submission_Date is more than 30 days ago from today, OR
#   - Status is Approved/Partially Approved AND Processing_Days > 30
DELAY_THRESHOLD_DAYS = 30

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Healthcare Revenue Cycle Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Data loading & validation
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading dataset…")
def load_and_validate(path: str) -> tuple[pd.DataFrame, list[str]]:
    """
    Load the CSV dataset and validate data quality.
    Returns (dataframe, list_of_warnings).
    Raises FileNotFoundError or ValueError on critical issues.
    """
    warnings: list[str] = []

    # --- File existence check ---
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n\n"
            "Please ensure 'data/claims_data.csv' exists in the project folder."
        )

    df = pd.read_csv(path)

    # --- Required columns check ---
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset is missing required columns: {missing_cols}\n\n"
            "Please verify the dataset schema matches the project specification."
        )

    # --- Date parsing ---
    for col in ("Submission_Date", "Settlement_Date"):
        df[col] = pd.to_datetime(df[col], errors="coerce")

    invalid_sub = df["Submission_Date"].isna().sum()
    if invalid_sub > 0:
        warnings.append(f"{invalid_sub} row(s) have invalid or missing Submission_Date and will be excluded from date-sensitive calculations.")

    # --- Remove rows with missing Submission_Date (critical field) ---
    df = df[df["Submission_Date"].notna()].copy()

    # --- Settlement before Submission check ---
    settled_mask = df["Settlement_Date"].notna()
    invalid_order = (df.loc[settled_mask, "Settlement_Date"] < df.loc[settled_mask, "Submission_Date"]).sum()
    if invalid_order > 0:
        warnings.append(f"{invalid_order} row(s) have Settlement_Date earlier than Submission_Date. These are flagged but retained.")

    # --- Numeric columns validation ---
    for col in ("Claim_Amount", "Approved_Amount", "Rejected_Amount", "Pending_Amount"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        neg = (df[col] < 0).sum()
        if neg > 0:
            warnings.append(f"{neg} row(s) have negative values in '{col}'. These will be treated as 0.")
            df[col] = df[col].clip(lower=0)

    # --- Approved amount greater than claim amount ---
    over_approved = (df["Approved_Amount"] > df["Claim_Amount"]).sum()
    if over_approved > 0:
        warnings.append(f"{over_approved} row(s) have Approved_Amount greater than Claim_Amount.")

    # --- Invalid case status values ---
    invalid_status = ~df["Case_Status"].isin(VALID_STATUSES)
    if invalid_status.sum() > 0:
        warnings.append(
            f"{invalid_status.sum()} row(s) have unrecognised Case_Status values: "
            f"{df.loc[invalid_status, 'Case_Status'].unique().tolist()}. These rows are retained."
        )

    # --- Missing values summary ---
    for col in ("Department", "TPA_or_Payer", "Patient_Type"):
        miss = df[col].isna().sum()
        if miss > 0:
            warnings.append(f"{miss} row(s) have missing values in '{col}'.")

    df["Query_or_Rejection_Reason"] = df["Query_or_Rejection_Reason"].fillna("None")

    # --- Derived columns ---
    today = pd.Timestamp(date.today())

    # Processing days: for settled cases use settlement date; for pending use today
    df["Processing_Days"] = np.where(
        df["Settlement_Date"].notna(),
        (df["Settlement_Date"] - df["Submission_Date"]).dt.days,
        (today - df["Submission_Date"]).dt.days,
    )
    df["Processing_Days"] = df["Processing_Days"].clip(lower=0)

    # Delayed flag
    df["Is_Delayed"] = np.where(
        df["Case_Status"] == "Pending",
        (today - df["Submission_Date"]).dt.days > DELAY_THRESHOLD_DAYS,
        df["Processing_Days"] > DELAY_THRESHOLD_DAYS,
    )

    df["Submission_Month"] = df["Submission_Date"].dt.to_period("M").astype(str)

    return df, warnings


# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------

def render_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.title("🔽 Filters")

    # Date range
    min_date = df["Submission_Date"].min().date()
    max_date = df["Submission_Date"].max().date()
    date_range = st.sidebar.date_input(
        "Submission Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range

    # Department
    dept_options = sorted(df["Department"].dropna().unique().tolist())
    departments = st.sidebar.multiselect("Department", dept_options, default=dept_options)

    # TPA / Payer
    tpa_options = sorted(df["TPA_or_Payer"].dropna().unique().tolist())
    tpas = st.sidebar.multiselect("TPA / Payer", tpa_options, default=tpa_options)

    # Patient type
    pt_options = sorted(df["Patient_Type"].dropna().unique().tolist())
    patient_types = st.sidebar.multiselect("Patient Type", pt_options, default=pt_options)

    # Case status
    status_options = sorted(df["Case_Status"].dropna().unique().tolist())
    statuses = st.sidebar.multiselect("Case Status", status_options, default=status_options)

    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"**Delayed-case assumption:** A case is flagged as delayed when "
        f"it has been Pending for > {DELAY_THRESHOLD_DAYS} days, or when a "
        f"settled case took > {DELAY_THRESHOLD_DAYS} days to process."
    )

    # Apply filters
    mask = (
        (df["Submission_Date"].dt.date >= start_date)
        & (df["Submission_Date"].dt.date <= end_date)
        & (df["Department"].isin(departments))
        & (df["TPA_or_Payer"].isin(tpas))
        & (df["Patient_Type"].isin(patient_types))
        & (df["Case_Status"].isin(statuses))
    )
    return df[mask].copy()


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def render_overview(df: pd.DataFrame) -> None:
    st.header("📊 Overview Dashboard")

    total_cases = len(df)
    total_claim = df["Claim_Amount"].sum()
    approved_amt = df["Approved_Amount"].sum()
    pending_amt = df["Pending_Amount"].sum()
    rejected_amt = df["Rejected_Amount"].sum()
    delayed_cases = int(df["Is_Delayed"].sum())
    avg_proc = df["Processing_Days"].mean() if total_cases > 0 else 0

    approved_cases = (df["Case_Status"] == "Approved").sum() + (df["Case_Status"] == "Partially Approved").sum()
    rejected_cases = (df["Case_Status"] == "Rejected").sum()
    approval_rate = (approved_cases / total_cases * 100) if total_cases > 0 else 0
    rejection_rate = (rejected_cases / total_cases * 100) if total_cases > 0 else 0

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Cases", f"{total_cases:,}")
    col2.metric("Total Claim Amount", f"₹{total_claim:,.0f}")
    col3.metric("Approved Amount", f"₹{approved_amt:,.0f}")
    col4.metric("Pending Amount", f"₹{pending_amt:,.0f}")
    col5.metric("Rejected Amount", f"₹{rejected_amt:,.0f}")

    col6, col7, col8, col9 = st.columns(4)
    col6.metric("Approval Rate", f"{approval_rate:.1f}%")
    col7.metric("Rejection Rate", f"{rejection_rate:.1f}%")
    col8.metric("Avg Processing Time", f"{avg_proc:.1f} days")
    col9.metric("Delayed Cases", f"{delayed_cases}", delta=None)

    st.markdown("---")

    row1col1, row1col2 = st.columns(2)

    # Cases by status — pie chart
    status_counts = df["Case_Status"].value_counts().reset_index()
    status_counts.columns = ["Case_Status", "Count"]
    fig_pie = px.pie(
        status_counts,
        names="Case_Status",
        values="Count",
        title="Cases by Status",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
    row1col1.plotly_chart(fig_pie, use_container_width=True)

    # Claim amount by status — bar chart
    amt_by_status = (
        df.groupby("Case_Status")[["Claim_Amount", "Approved_Amount", "Rejected_Amount", "Pending_Amount"]]
        .sum()
        .reset_index()
    )
    fig_bar = px.bar(
        amt_by_status.melt(id_vars="Case_Status", var_name="Amount_Type", value_name="Amount"),
        x="Case_Status",
        y="Amount",
        color="Amount_Type",
        barmode="group",
        title="Claim Amounts by Status",
        labels={"Amount": "Amount (₹)", "Case_Status": "Case Status"},
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    row1col2.plotly_chart(fig_bar, use_container_width=True)

    # Monthly case volume — line chart
    monthly = df.groupby("Submission_Month").size().reset_index(name="Case_Count")
    monthly = monthly.sort_values("Submission_Month")
    fig_line = px.line(
        monthly,
        x="Submission_Month",
        y="Case_Count",
        title="Monthly Case Volume",
        markers=True,
        labels={"Submission_Month": "Month", "Case_Count": "Number of Cases"},
    )
    fig_line.update_xaxes(tickangle=45)
    st.plotly_chart(fig_line, use_container_width=True)


def render_claim_records(df: pd.DataFrame) -> None:
    st.header("📋 Claim Records")

    search_id = st.text_input("Search by Case ID", placeholder="e.g. C001").strip().upper()

    display_df = df.copy()
    if search_id:
        display_df = display_df[display_df["Case_ID"].str.upper().str.contains(search_id, na=False)]
        if display_df.empty:
            st.warning(f"No records found for Case ID containing '{search_id}'.")

    cols_to_show = [
        "Case_ID", "Patient_Type", "Department", "TPA_or_Payer", "Service_Type",
        "Claim_Amount", "Approved_Amount", "Rejected_Amount", "Pending_Amount",
        "Case_Status", "Submission_Date", "Settlement_Date", "Processing_Days", "Is_Delayed",
        "Query_or_Rejection_Reason",
    ]
    cols_to_show = [c for c in cols_to_show if c in display_df.columns]

    st.caption(f"Showing {len(display_df):,} record(s).")

    display_df_view = display_df[cols_to_show].copy()
    display_df_view["Submission_Date"] = display_df_view["Submission_Date"].dt.strftime("%Y-%m-%d")
    display_df_view["Settlement_Date"] = display_df_view["Settlement_Date"].dt.strftime("%Y-%m-%d").replace("NaT", "")

    st.dataframe(display_df_view, use_container_width=True, hide_index=True)


def render_tpa_analysis(df: pd.DataFrame) -> None:
    st.header("🏦 TPA Analysis")

    if df.empty:
        st.info("No data available for the selected filters.")
        return

    # Summary table
    tpa_summary = df.groupby("TPA_or_Payer").agg(
        Total_Cases=("Case_ID", "count"),
        Total_Claim=("Claim_Amount", "sum"),
        Total_Approved=("Approved_Amount", "sum"),
        Total_Rejected=("Rejected_Amount", "sum"),
        Total_Pending=("Pending_Amount", "sum"),
        Avg_Processing_Days=("Processing_Days", "mean"),
        Delayed_Cases=("Is_Delayed", "sum"),
    ).reset_index()

    tpa_summary["Approval_Rate_%"] = (
        tpa_summary["Total_Approved"] / tpa_summary["Total_Claim"].replace(0, np.nan) * 100
    ).round(1)
    tpa_summary["Avg_Processing_Days"] = tpa_summary["Avg_Processing_Days"].round(1)

    st.subheader("TPA Performance Summary")
    st.dataframe(tpa_summary, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)

    # Average processing time by TPA
    fig_tpa_proc = px.bar(
        tpa_summary.sort_values("Avg_Processing_Days", ascending=False),
        x="TPA_or_Payer",
        y="Avg_Processing_Days",
        title="Average Processing Time by TPA / Payer",
        labels={"TPA_or_Payer": "TPA / Payer", "Avg_Processing_Days": "Avg Processing Days"},
        color="Avg_Processing_Days",
        color_continuous_scale="RdYlGn_r",
        text="Avg_Processing_Days",
    )
    fig_tpa_proc.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    col1.plotly_chart(fig_tpa_proc, use_container_width=True)

    # Claim amount by TPA — grouped bar
    fig_tpa_amt = px.bar(
        tpa_summary,
        x="TPA_or_Payer",
        y=["Total_Claim", "Total_Approved", "Total_Rejected", "Total_Pending"],
        title="Claim Amounts by TPA / Payer",
        labels={"value": "Amount (₹)", "TPA_or_Payer": "TPA / Payer"},
        barmode="group",
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    col2.plotly_chart(fig_tpa_amt, use_container_width=True)

    # Approval rate by TPA
    fig_approval = px.bar(
        tpa_summary.sort_values("Approval_Rate_%", ascending=True),
        x="Approval_Rate_%",
        y="TPA_or_Payer",
        orientation="h",
        title="Approval Rate (%) by TPA / Payer",
        labels={"Approval_Rate_%": "Approval Rate (%)", "TPA_or_Payer": "TPA / Payer"},
        color="Approval_Rate_%",
        color_continuous_scale="Blues",
        text="Approval_Rate_%",
    )
    fig_approval.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    st.plotly_chart(fig_approval, use_container_width=True)


def render_department_analysis(df: pd.DataFrame) -> None:
    st.header("🏥 Department Analysis")

    if df.empty:
        st.info("No data available for the selected filters.")
        return

    dept_summary = df.groupby("Department").agg(
        Total_Cases=("Case_ID", "count"),
        Total_Claim=("Claim_Amount", "sum"),
        Total_Approved=("Approved_Amount", "sum"),
        Total_Rejected=("Rejected_Amount", "sum"),
        Total_Pending=("Pending_Amount", "sum"),
        Avg_Processing_Days=("Processing_Days", "mean"),
        Delayed_Cases=("Is_Delayed", "sum"),
    ).reset_index()

    dept_summary["Approval_Rate_%"] = (
        dept_summary["Total_Approved"] / dept_summary["Total_Claim"].replace(0, np.nan) * 100
    ).round(1)
    dept_summary["Avg_Processing_Days"] = dept_summary["Avg_Processing_Days"].round(1)

    st.subheader("Department Performance Summary")
    st.dataframe(dept_summary, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)

    # Pending amount by department
    fig_pend = px.bar(
        dept_summary.sort_values("Total_Pending", ascending=False),
        x="Department",
        y="Total_Pending",
        title="Pending Amount by Department",
        labels={"Total_Pending": "Pending Amount (₹)", "Department": "Department"},
        color="Total_Pending",
        color_continuous_scale="OrRd",
        text="Total_Pending",
    )
    fig_pend.update_traces(texttemplate="₹%{text:,.0f}", textposition="outside")
    col1.plotly_chart(fig_pend, use_container_width=True)

    # Average processing time by department
    fig_dept_proc = px.bar(
        dept_summary.sort_values("Avg_Processing_Days", ascending=False),
        x="Department",
        y="Avg_Processing_Days",
        title="Average Processing Time by Department",
        labels={"Avg_Processing_Days": "Avg Processing Days", "Department": "Department"},
        color="Avg_Processing_Days",
        color_continuous_scale="RdYlGn_r",
        text="Avg_Processing_Days",
    )
    fig_dept_proc.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    col2.plotly_chart(fig_dept_proc, use_container_width=True)

    # Delayed cases by department
    fig_delay = px.bar(
        dept_summary.sort_values("Delayed_Cases", ascending=False),
        x="Department",
        y="Delayed_Cases",
        title="Delayed Cases by Department",
        labels={"Delayed_Cases": "Number of Delayed Cases", "Department": "Department"},
        color="Delayed_Cases",
        color_continuous_scale="Reds",
        text="Delayed_Cases",
    )
    fig_delay.update_traces(texttemplate="%{text}", textposition="outside")
    st.plotly_chart(fig_delay, use_container_width=True)


def render_process_performance(df: pd.DataFrame) -> None:
    st.header("⚙️ Process Performance")

    if df.empty:
        st.info("No data available for the selected filters.")
        return

    col1, col2 = st.columns(2)

    # Processing time distribution
    settled_df = df[df["Settlement_Date"].notna()].copy()
    if not settled_df.empty:
        fig_hist = px.histogram(
            settled_df,
            x="Processing_Days",
            nbins=20,
            title="Distribution of Processing Time (Settled Cases)",
            labels={"Processing_Days": "Processing Days", "count": "Number of Cases"},
            color_discrete_sequence=["#3b82d4"],
        )
        fig_hist.add_vline(
            x=DELAY_THRESHOLD_DAYS, line_dash="dash", line_color="red",
            annotation_text=f"Delay threshold ({DELAY_THRESHOLD_DAYS}d)",
        )
        col1.plotly_chart(fig_hist, use_container_width=True)
    else:
        col1.info("No settled cases to display processing time distribution.")

    # Processing time by service type
    svc_proc = df.groupby("Service_Type")["Processing_Days"].mean().reset_index()
    svc_proc.columns = ["Service_Type", "Avg_Processing_Days"]
    svc_proc = svc_proc.sort_values("Avg_Processing_Days", ascending=False)
    fig_svc = px.bar(
        svc_proc,
        x="Service_Type",
        y="Avg_Processing_Days",
        title="Avg Processing Time by Service Type",
        labels={"Avg_Processing_Days": "Avg Processing Days", "Service_Type": "Service Type"},
        color="Avg_Processing_Days",
        color_continuous_scale="Teal",
        text="Avg_Processing_Days",
    )
    fig_svc.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    col2.plotly_chart(fig_svc, use_container_width=True)

    # Monthly claim volume and approval trends
    monthly = df.groupby("Submission_Month").agg(
        Total_Cases=("Case_ID", "count"),
        Approved_Cases=("Case_Status", lambda x: (x.isin(["Approved", "Partially Approved"])).sum()),
        Delayed_Cases=("Is_Delayed", "sum"),
    ).reset_index().sort_values("Submission_Month")

    fig_monthly_trend = go.Figure()
    fig_monthly_trend.add_trace(go.Scatter(
        x=monthly["Submission_Month"], y=monthly["Total_Cases"],
        mode="lines+markers", name="Total Cases",
    ))
    fig_monthly_trend.add_trace(go.Scatter(
        x=monthly["Submission_Month"], y=monthly["Approved_Cases"],
        mode="lines+markers", name="Approved Cases",
    ))
    fig_monthly_trend.add_trace(go.Scatter(
        x=monthly["Submission_Month"], y=monthly["Delayed_Cases"],
        mode="lines+markers", name="Delayed Cases", line=dict(dash="dot", color="red"),
    ))
    fig_monthly_trend.update_layout(
        title="Monthly Case Volume & Approval Trend",
        xaxis_title="Month",
        yaxis_title="Number of Cases",
        xaxis_tickangle=45,
        legend_title="Metric",
    )
    st.plotly_chart(fig_monthly_trend, use_container_width=True)

    # Delayed cases detail
    st.subheader("Delayed Cases Detail")
    delayed_df = df[df["Is_Delayed"]].copy()
    st.caption(f"{len(delayed_df)} delayed case(s) identified.")
    if not delayed_df.empty:
        view_cols = ["Case_ID", "Department", "TPA_or_Payer", "Case_Status", "Processing_Days",
                     "Submission_Date", "Claim_Amount", "Pending_Amount"]
        view_cols = [c for c in view_cols if c in delayed_df.columns]
        delayed_view = delayed_df[view_cols].copy()
        delayed_view["Submission_Date"] = delayed_view["Submission_Date"].dt.strftime("%Y-%m-%d")
        st.dataframe(delayed_view.sort_values("Processing_Days", ascending=False), use_container_width=True, hide_index=True)
    else:
        st.success("No delayed cases in the current selection.")


def render_query_rejection_analysis(df: pd.DataFrame) -> None:
    st.header("❓ Query and Rejection Analysis")

    if df.empty:
        st.info("No data available for the selected filters.")
        return

    # Filter to cases with a meaningful reason
    reason_df = df[df["Query_or_Rejection_Reason"].notna() & (df["Query_or_Rejection_Reason"] != "None")].copy()

    if reason_df.empty:
        st.info("No query or rejection reasons available for the selected filters.")
        return

    col1, col2 = st.columns(2)

    # Most common reasons
    reason_counts = reason_df["Query_or_Rejection_Reason"].value_counts().reset_index()
    reason_counts.columns = ["Reason", "Count"]

    fig_reason = px.bar(
        reason_counts,
        x="Count",
        y="Reason",
        orientation="h",
        title="Most Common Query / Rejection Reasons",
        labels={"Count": "Number of Cases", "Reason": "Reason"},
        color="Count",
        color_continuous_scale="Reds",
        text="Count",
    )
    fig_reason.update_traces(texttemplate="%{text}", textposition="outside")
    col1.plotly_chart(fig_reason, use_container_width=True)

    # Reasons by department
    reason_dept = reason_df.groupby(["Department", "Query_or_Rejection_Reason"]).size().reset_index(name="Count")
    fig_reason_dept = px.bar(
        reason_dept,
        x="Department",
        y="Count",
        color="Query_or_Rejection_Reason",
        title="Query / Rejection Reasons by Department",
        labels={"Count": "Number of Cases", "Department": "Department"},
        barmode="stack",
        color_discrete_sequence=px.colors.qualitative.Set3,
    )
    col2.plotly_chart(fig_reason_dept, use_container_width=True)

    # Reasons by TPA
    reason_tpa = reason_df.groupby(["TPA_or_Payer", "Query_or_Rejection_Reason"]).size().reset_index(name="Count")
    fig_reason_tpa = px.bar(
        reason_tpa,
        x="TPA_or_Payer",
        y="Count",
        color="Query_or_Rejection_Reason",
        title="Query / Rejection Reasons by TPA / Payer",
        labels={"Count": "Number of Cases", "TPA_or_Payer": "TPA / Payer"},
        barmode="stack",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    st.plotly_chart(fig_reason_tpa, use_container_width=True)

    # Detailed table
    st.subheader("Cases with Query or Rejection Reason")
    view_cols = ["Case_ID", "Department", "TPA_or_Payer", "Case_Status",
                 "Claim_Amount", "Query_or_Rejection_Reason", "Processing_Days"]
    view_cols = [c for c in view_cols if c in reason_df.columns]
    st.dataframe(reason_df[view_cols], use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Prediction model
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Training prediction model…")
def train_rejection_model(df: pd.DataFrame):
    """
    Train a Random Forest classifier to predict whether a claim will be
    Rejected (1) or Not Rejected (0).
    Features: Department, TPA_or_Payer, Patient_Type, Service_Type, Claim_Amount
    Returns: (model, label_encoders, feature_names, report_dict, cm, classes)
    """
    model_df = df.copy()

    # Binary target: Rejected = 1, everything else = 0
    model_df["Target"] = (model_df["Case_Status"] == "Rejected").astype(int)

    cat_cols = ["Department", "TPA_or_Payer", "Patient_Type", "Service_Type"]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        model_df[col + "_enc"] = le.fit_transform(model_df[col].astype(str))
        encoders[col] = le

    feature_cols = [c + "_enc" for c in cat_cols] + ["Claim_Amount"]
    X = model_df[feature_cols]
    y = model_df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y if y.sum() > 1 else None
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    return clf, encoders, cat_cols, feature_cols, report, cm


def render_prediction(df: pd.DataFrame) -> None:
    st.header("🤖 Claim Rejection Predictor")
    st.caption(
        "Uses a Random Forest model trained on historical claims to predict whether "
        "a new claim is likely to be **Rejected** or **Not Rejected**."
    )

    # Train model
    clf, encoders, cat_cols, feature_cols, report, cm = train_rejection_model(df)

    # ── Model performance ──────────────────────────────────────────────────
    with st.expander("📈 Model Performance (on test split)", expanded=False):
        col1, col2, col3 = st.columns(3)
        col1.metric("Precision (Rejected)", f"{report.get('1', {}).get('precision', 0):.0%}")
        col2.metric("Recall (Rejected)",    f"{report.get('1', {}).get('recall', 0):.0%}")
        col3.metric("Accuracy",             f"{report.get('accuracy', 0):.0%}")

        st.markdown("**Confusion Matrix**")
        cm_df = pd.DataFrame(
            cm,
            index=["Actual: Not Rejected", "Actual: Rejected"],
            columns=["Predicted: Not Rejected", "Predicted: Rejected"],
        )
        st.dataframe(cm_df, use_container_width=True)

        # Feature importance
        importances = pd.DataFrame({
            "Feature": [c.replace("_enc", "") for c in feature_cols],
            "Importance": clf.feature_importances_,
        }).sort_values("Importance", ascending=True)

        fig_imp = px.bar(
            importances,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Feature Importance",
            color="Importance",
            color_continuous_scale="Blues",
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("---")
    st.subheader("🔍 Predict a New Claim")

    # ── Input form ─────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    dept_options     = sorted(df["Department"].dropna().unique().tolist())
    tpa_options      = sorted(df["TPA_or_Payer"].dropna().unique().tolist())
    pt_options       = sorted(df["Patient_Type"].dropna().unique().tolist())
    service_options  = sorted(df["Service_Type"].dropna().unique().tolist())

    with col1:
        inp_dept    = st.selectbox("Department",      dept_options)
        inp_tpa     = st.selectbox("TPA / Payer",     tpa_options)

    with col2:
        inp_pt      = st.selectbox("Patient Type",    pt_options)
        inp_service = st.selectbox("Service Type",    service_options)

    inp_amount = st.number_input(
        "Claim Amount (₹)", min_value=1000, max_value=10_000_000,
        value=50000, step=5000
    )

    if st.button("Predict", type="primary"):
        row = {
            "Department":   inp_dept,
            "TPA_or_Payer": inp_tpa,
            "Patient_Type": inp_pt,
            "Service_Type": inp_service,
            "Claim_Amount": inp_amount,
        }

        # Encode inputs
        encoded = []
        for col in cat_cols:
            le = encoders[col]
            val = row[col]
            if val in le.classes_:
                encoded.append(le.transform([val])[0])
            else:
                encoded.append(0)
        encoded.append(inp_amount)

        input_array = np.array(encoded).reshape(1, -1)
        prediction  = clf.predict(input_array)[0]
        probability = clf.predict_proba(input_array)[0]

        rejected_prob     = probability[1] if len(probability) > 1 else probability[0]
        not_rejected_prob = 1 - rejected_prob

        st.markdown("---")
        if prediction == 1:
            st.error(f"⚠️ **Prediction: HIGH REJECTION RISK** — {rejected_prob:.0%} probability of rejection")
        else:
            st.success(f"✅ **Prediction: LIKELY TO BE APPROVED** — {not_rejected_prob:.0%} probability of approval")

        # Probability bar
        prob_df = pd.DataFrame({
            "Outcome":      ["Not Rejected", "Rejected"],
            "Probability":  [not_rejected_prob, rejected_prob],
        })
        fig_prob = px.bar(
            prob_df,
            x="Outcome",
            y="Probability",
            color="Outcome",
            color_discrete_map={"Not Rejected": "#22c55e", "Rejected": "#ef4444"},
            title="Prediction Probability",
            text_auto=".0%",
        )
        fig_prob.update_layout(yaxis_tickformat=".0%", showlegend=False)
        st.plotly_chart(fig_prob, use_container_width=True)

        # Similar historical claims
        st.subheader("📋 Similar Historical Claims")
        similar = df[
            (df["Department"]   == inp_dept) &
            (df["TPA_or_Payer"] == inp_tpa)  &
            (df["Patient_Type"] == inp_pt)
        ][["Case_ID", "Service_Type", "Claim_Amount", "Case_Status", "Query_or_Rejection_Reason", "Processing_Days"]]
        if similar.empty:
            st.info("No historical claims found with the same Department + TPA + Patient Type combination.")
        else:
            st.caption(f"{len(similar)} similar claim(s) found.")
            st.dataframe(similar, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("🏥 Healthcare Revenue Cycle & TPA Process Analytics Dashboard")
    st.caption(
        "A centralized view of revenue-cycle and TPA performance to identify delays, "
        "prioritize follow-up, and support process-improvement decisions."
    )

    # --- Load data ---
    try:
        raw_df, warnings = load_and_validate(DATASET_PATH)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    # Show data-quality warnings (non-blocking)
    if warnings:
        with st.expander("⚠️ Data Quality Notices", expanded=False):
            for w in warnings:
                st.warning(w)

    # --- Sidebar filters ---
    filtered_df = render_sidebar(raw_df)

    if filtered_df.empty:
        st.warning("No records match the selected filters. Please adjust the filter criteria in the sidebar.")
        st.stop()

    # --- Navigation ---
    sections = [
        "Overview Dashboard",
        "Claim Records",
        "TPA Analysis",
        "Department Analysis",
        "Process Performance",
        "Query and Rejection Analysis",
        "Claim Rejection Predictor",
    ]
    section = st.selectbox("Navigate to section", sections, index=0, label_visibility="collapsed")

    st.divider()

    if section == "Overview Dashboard":
        render_overview(filtered_df)
    elif section == "Claim Records":
        render_claim_records(filtered_df)
    elif section == "TPA Analysis":
        render_tpa_analysis(filtered_df)
    elif section == "Department Analysis":
        render_department_analysis(filtered_df)
    elif section == "Process Performance":
        render_process_performance(filtered_df)
    elif section == "Query and Rejection Analysis":
        render_query_rejection_analysis(filtered_df)
    elif section == "Claim Rejection Predictor":
        render_prediction(raw_df)


if __name__ == "__main__":
    main()
