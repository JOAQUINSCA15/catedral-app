# Shift Scheduler App (Streamlit + PuLP)

A lightweight weekly shift scheduling tool. Upload a **7×15 demand CSV** (Mon..Sun × 15 hourly slots),
edit your **staff list** directly in the app (add/remove rows), set a **max deviation per slot**, and solve.

## Folder structure
```text
shift_scheduler_app/
├── streamlit_app.py           # Streamlit UI (file upload, staff editor, parameters, results & downloads)
├── optimizer.py               # PuLP model (variables, constraints, objective, CBC solve)
├── requirements.txt           # Dependencies
├── README.md                  # This guide
└── sales_demand_template.csv  # 7×15 template (no header)
```

## Quick start
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Demand CSV (sales_demand_template.csv)
- **Shape**: 7 rows × 15 columns (no header).
- **Rows**: 1=Monday, ..., 7=Sunday.
- **Columns**: 15 hourly slots (10-11, 11-12, ..., 23-00, 00-01).
- **Values**: headcount demand per slot (you can use sales/100 as an approximation).

## Staff table (editable in UI)
Columns:
- `name` (string, unique)
- `min_week_hours` (float/int)
- `max_week_hours` (float/int)
- `contract_hours` (float/int, info only)
- `weekend_only` (bool) — if True, worker can only be scheduled on Fri/Sat/Sun

You can **Upload** an existing staff CSV or **Download** the current table for reuse.

## Model (high level)
- Decision: assign at most one shift per worker per day; each shift is 4–8 hours.
- Coverage: minimize total deviation (under + over) with a **cap** per slot.
- Per-worker weekly hours in [min_week_hours, max_week_hours].
- Weekend-only workers: disallow Mon–Thu assignments.

> Notes: Consecutive rest days, late-to-early gap, or other business rules can be added in `optimizer.py` in the same style.

## Troubleshooting
- If CBC is missing, upgrade PuLP (`pip install -U pulp`) or install COIN-OR CBC in your system.
- Increase the solver time limit if the instance is large.
