# Agent Instructions — Healthcare Revenue Cycle and TPA Process Analytics Dashboard

## Purpose of This Document

This file documents the design decisions, restrictions, assumptions, and structural expectations of the project for any AI agent (such as IBM Bob) that maintains, extends, or regenerates code for this project.

---

## Project Purpose

Build a **Python-only Streamlit dashboard** that gives healthcare finance and billing teams a centralized view of:

- Revenue-cycle claim status and amounts (approved, pending, rejected).
- TPA (Third-Party Administrator) performance comparison.
- Department-level performance comparison.
- Delayed case identification.
- Query and rejection reason analysis.

The dashboard is intended to reduce manual reconciliation effort, highlight bottlenecks, and support process-improvement decisions.

---

## Technology Restrictions

| Restriction | Rationale |
|---|---|
| **Python only** | All logic, rendering, and data processing must be in Python. |
| **No HTML** | Do not write raw HTML strings, `st.markdown(..., unsafe_allow_html=True)`, or HTML template files. |
| **No CSS** | Do not write inline or external CSS styles. Use Streamlit's default styling only. |
| **No JavaScript** | Do not write JS event handlers, scripts, or JS-based frameworks. |
| **No Flask / Django / React / Node.js** | Streamlit is the only permitted application framework. |
| **No scikit-learn or ML models** | No predictive models, classifiers, or regressors. |
| **No external AI APIs** | No OpenAI, Gemini, watsonx, or any API key–based AI service inside the application. |
| **No chatbot feature** | The application must not include a chat interface or AI-generated response feature. |
| **No auto-generated dummy data** | If `data/claims_data.csv` is missing, display an error message and stop. Do not silently generate replacement data. |

---

## Dataset Fields

The dataset is located at `data/claims_data.csv` (relative to `app.py`).

| Column | Type | Description |
|---|---|---|
| Case_ID | string | Unique case identifier (e.g. C001) |
| Patient_Type | string | Inpatient / Outpatient / Daycare |
| Department | string | Clinical department name |
| TPA_or_Payer | string | TPA or insurance company name |
| Service_Type | string | Type of medical service |
| Claim_Amount | float | Total claim amount in ₹ |
| Approved_Amount | float | Approved amount in ₹ |
| Rejected_Amount | float | Rejected amount in ₹ |
| Pending_Amount | float | Pending amount in ₹ |
| Submission_Date | date (YYYY-MM-DD) | Date claim was submitted |
| Settlement_Date | date (YYYY-MM-DD) | Date claim was settled; blank if open |
| Case_Status | string | Approved / Rejected / Pending / Partially Approved |
| Query_or_Rejection_Reason | string | Reason if queried or rejected; empty if N/A |

All data is synthetic. Do not load real patient data.

---

## KPI Definitions

| KPI | Formula |
|---|---|
| Total Cases | Count of all rows in the filtered dataset |
| Total Claim Amount | Sum of Claim_Amount |
| Approved Amount | Sum of Approved_Amount |
| Pending Amount | Sum of Pending_Amount |
| Rejected Amount | Sum of Rejected_Amount |
| Approval Rate | (Count where Case_Status ∈ {Approved, Partially Approved}) ÷ Total Cases × 100 |
| Rejection Rate | (Count where Case_Status = Rejected) ÷ Total Cases × 100 |
| Processing Days | Settlement_Date − Submission_Date for settled; today − Submission_Date for Pending |
| Average Processing Time | Mean of Processing_Days across the filtered dataset |
| Delayed Cases | Count of rows where Is_Delayed = True |

---

## Delayed-Case Assumption

**Constant:** `DELAY_THRESHOLD_DAYS = 30`

A case is flagged as **Is_Delayed = True** when:

1. `Case_Status == "Pending"` **AND** `(today − Submission_Date) > 30 days`, **OR**
2. `Case_Status ∈ {Approved, Partially Approved, Rejected}` **AND** `Processing_Days > 30`

This assumption is clearly stated in the sidebar caption and in README.md. Adjust `DELAY_THRESHOLD_DAYS` in `app.py` to match your organization's SLA.

---

## Expected Application Structure

```
healthcare_revenue_cycle_project/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python package dependencies
├── README.md                 # Project documentation
├── agent_instructions.md     # This file — AI agent context
└── data/
    └── claims_data.csv       # Synthetic claims dataset
```

`app.py` is self-contained. All logic is in a single file to keep the project simple and auditable.

---

## Section Map

| Section | Function in app.py | Key Content |
|---|---|---|
| Overview Dashboard | `render_overview()` | KPI metrics, status pie, amounts bar, monthly line |
| Claim Records | `render_claim_records()` | Searchable, filterable dataframe |
| TPA Analysis | `render_tpa_analysis()` | TPA summary table, processing time bar, approval rate |
| Department Analysis | `render_department_analysis()` | Dept summary, pending amounts, processing time, delayed |
| Process Performance | `render_process_performance()` | Processing time histogram, monthly trend, delayed detail |
| Query and Rejection Analysis | `render_query_rejection_analysis()` | Reason frequency bar, by-dept stack, by-TPA stack |

---

## Validation Checklist for Any Agent Editing This Project

Before submitting a change, verify:

- [ ] Application runs with `streamlit run app.py` from the project directory.
- [ ] Dataset is loaded from `data/claims_data.csv` using a relative path via `os.path.dirname(__file__)`.
- [ ] Missing dataset triggers `st.error()` and `st.stop()` — not silent fallback data.
- [ ] No `st.markdown(..., unsafe_allow_html=True)` calls.
- [ ] No `<html>`, `<style>`, `<script>`, or CSS class strings anywhere in the Python code.
- [ ] No imports from `sklearn`, `tensorflow`, `torch`, `openai`, `anthropic`, or similar.
- [ ] All Plotly charts use `st.plotly_chart(..., use_container_width=True)`.
- [ ] All dataframes use `st.dataframe(..., use_container_width=True, hide_index=True)`.
- [ ] `requirements.txt` lists all and only the packages actually imported.
- [ ] Delayed-case assumption is documented in sidebar caption and README.md.
- [ ] All paths are relative (no hard-coded absolute paths).
