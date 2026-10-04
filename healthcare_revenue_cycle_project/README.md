# Healthcare Revenue Cycle and TPA Process Analytics Dashboard

## Project Description

A Python-only Streamlit dashboard that provides a centralized view of healthcare revenue-cycle activities and Third-Party Administrator (TPA) performance. The dashboard enables hospital finance and billing teams to monitor claim status, track processing time, identify delayed cases, and compare TPA and department performance — all from a single interface.

---

## Problem Statement

Hospitals may find it difficult to monitor revenue-cycle activities through separate records. This can make it harder to track claim status, processing time, pending amounts, and delayed cases. The proposed dashboard provides a centralized view of revenue-cycle and TPA performance to help users identify delays and prioritize follow-up.

---

## Project Objective

Develop a Streamlit dashboard that:

- Monitors revenue-cycle activities.
- Tracks claim status and processing time.
- Displays pending, approved, and rejected claim amounts.
- Compares TPA and department performance.
- Identifies delayed cases.
- Shows common query and rejection reasons.
- Supports process-improvement decisions.

---

## Scope

- Inpatient, outpatient, and daycare cases.
- Revenue-cycle KPI tracking (approval rate, rejection rate, pending amounts).
- TPA-level and department-level performance comparison.
- Delayed case identification based on a documented threshold.
- Query and rejection reason analysis.
- Fully local, no external API or internet connection required.

---

## Technology Used

| Component | Technology |
|---|---|
| Application framework | Streamlit |
| Data manipulation | pandas, numpy |
| Visualisation | Plotly |
| Dataset format | CSV |
| Language | Python 3.10+ |

No HTML, CSS, JavaScript, machine-learning models, or external AI APIs are used.

---

## Dataset Description

**File:** `data/claims_data.csv`  
**Records:** 130 synthetic, anonymized healthcare claims (January 2024 – January 2025)

| Column | Description |
|---|---|
| Case_ID | Unique case identifier |
| Patient_Type | Inpatient / Outpatient / Daycare |
| Department | Clinical department (Cardiology, Neurology, etc.) |
| TPA_or_Payer | Third-party administrator or insurance payer |
| Service_Type | Type of service (Surgery, ICU, Consultation, etc.) |
| Claim_Amount | Total amount claimed (₹) |
| Approved_Amount | Amount approved by the TPA (₹) |
| Rejected_Amount | Amount rejected by the TPA (₹) |
| Pending_Amount | Amount still pending settlement (₹) |
| Submission_Date | Date the claim was submitted |
| Settlement_Date | Date the claim was settled (blank if still open) |
| Case_Status | Approved / Rejected / Pending / Partially Approved |
| Query_or_Rejection_Reason | Reason for query or rejection (if applicable) |

All data is synthetic and does not represent any real patient or hospital.

---

## Derived Calculations

| KPI | Calculation |
|---|---|
| Processing Days | Settlement_Date − Submission_Date (or today if still Pending) |
| Approval Rate | (Approved + Partially Approved cases) ÷ Total cases × 100 |
| Rejection Rate | Rejected cases ÷ Total cases × 100 |
| Delayed Case | Pending for > 30 days **OR** settled but took > 30 days to process |

**Delayed-Case Assumption:** A processing threshold of **30 calendar days** is used. Cases pending beyond 30 days or settled cases whose turnaround exceeded 30 days are flagged as delayed. This threshold can be adjusted in `app.py` via the `DELAY_THRESHOLD_DAYS` constant.

---

## Installation Instructions

### Prerequisites

- Python 3.10 or higher
- pip

### Step 1 — Clone or download the project

```bash
git clone https://github.com/<your-username>/healthcare-revenue-cycle-dashboard.git
cd healthcare-revenue-cycle-dashboard/healthcare_revenue_cycle_project
```

Or extract the ZIP and navigate into `healthcare_revenue_cycle_project/`.

### Step 2 — Create and activate a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

---

## Local Run Instructions

```bash
streamlit run app.py
```

The application will open automatically at `http://localhost:8501`.

**Important:** Run this command from inside the `healthcare_revenue_cycle_project/` directory so that the relative path to `data/claims_data.csv` resolves correctly.

---

## GitHub Instructions

```bash
# Initialize repository
git init
git add .
git commit -m "Initial commit – Healthcare Revenue Cycle Dashboard"

# Link to remote repository
git remote add origin https://github.com/<your-username>/healthcare-revenue-cycle-dashboard.git
git push -u origin main
```

Ensure the `data/claims_data.csv` file is included in the commit. Do not commit any real patient data.

---

## Streamlit Deployment Instructions

1. Push the project to a public GitHub repository.
2. Go to [https://share.streamlit.io](https://share.streamlit.io) and sign in.
3. Click **New app**.
4. Set the repository, branch (`main`), and main file path:
   ```
   healthcare_revenue_cycle_project/app.py
   ```
5. Click **Deploy**.

Streamlit Community Cloud reads `requirements.txt` automatically and installs all dependencies.

---

## Data Privacy Note

All data in `data/claims_data.csv` is **fully synthetic and anonymized**. No real patient names, identifiers, hospital names, or financial records are included. This project complies with data-privacy best practices. Do not replace the synthetic dataset with real patient data without appropriate de-identification and compliance review.

---

## Project Limitations

- The dashboard is read-only; it does not support data entry or updates.
- The delayed-case threshold (30 days) is fixed in the source code. Adjust `DELAY_THRESHOLD_DAYS` in `app.py` if your organization uses a different SLA.
- The dataset is a static CSV file. For production use, integrate with a live database or claims management system.
- No predictive analytics or machine-learning models are included by design.
- Currency is displayed in Indian Rupees (₹). Modify the formatting string in `app.py` for other currencies.
