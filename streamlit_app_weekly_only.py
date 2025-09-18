import streamlit as st

# --- Guarded third-party imports with helpful messages ---
try:
    import pandas as pd
except Exception as e:
    st.error("Failed to import **pandas**. Please ensure it is installed.")
    st.code("pip install pandas>=2.0")
    st.stop()

try:
    import numpy as np
except Exception as e:
    st.error("Failed to import **numpy**. Please ensure it is installed.")
    st.code("pip install numpy")
    st.stop()

import inspect
from io import BytesIO
import sys, os
import math

# ------------------ Defaults ------------------
DEFAULT_STAFF = [
    {"name":"Ana",           "min_week_hours":30},
    {"name":"Vanessa_M",    "min_week_hours":25},
    {"name":"Ines",         "min_week_hours":30},
    {"name":"Yuliia",       "min_week_hours":20},
    {"name":"Giulia",       "min_week_hours":25},
    {"name":"Tomas",        "min_week_hours":30},
    {"name":"Ana_Bernabe",  "min_week_hours":25},
    {"name":"Raul_Calero",  "min_week_hours":25},
    {"name":"Jose_Antonio", "min_week_hours":20},
    {"name":"Haely",        "min_week_hours":25},
]
def _round_half(x): return math.floor(x*2+0.5)/2.0
for r in DEFAULT_STAFF:
    min_h = r["min_week_hours"]
    r["max_week_hours"] = _round_half(min(40.0, min_h*1.3))
    r["contract_hours"] = min_h
    r["weekend_only"] = False

# --- Import optimizer module robustly ---
opt_mod = None
opt_import_error = None
try:
    import optimizer as _opt_mod
    opt_mod = _opt_mod
except Exception as e1:
    _here = os.path.dirname(__file__)
    if _here and _here not in sys.path:
        sys.path.insert(0, _here)
    try:
        import optimizer as _opt_mod
        opt_mod = _opt_mod
    except Exception as e2:
        opt_import_error = (e1, e2)

def debug_import_error():
    st.error("Failed to import `optimizer.py`. Please ensure it is in the same folder as this file.")
    if opt_import_error is not None:
        e1, e2 = opt_import_error
        st.code(f"""
Original error 1: {type(e1).__name__}: {e1}
Retry error     : {type(e2).__name__}: {e2}
Working dir     : {os.getcwd()}
__file__        : {__file__}
sys.path[0..3]  : {sys.path[:4]}
Folder contents : {os.listdir(os.path.dirname(__file__)) if os.path.dirname(__file__) else 'N/A'}
        """)

def build_shift_set_fallback(T, min_len=4, max_len=8):
    S = []
    for s in T:
        for e in T:
            L = e - s + 1
            if min_len <= L <= max_len:
                S.append((s,e))
    return S

def adapt_to_user_optimizer(demand_df, staff_df, max_dev):
    """
    If optimizer has `build_and_solve_shift_model`, adapt inputs accordingly and call it.
    Returns a normalized dict if successful, else None.
    """
    if opt_mod is None or not hasattr(opt_mod, "build_and_solve_shift_model"):
        return None

    # Sets
    W = list(staff_df["name"])
    D = list(range(1, 8))
    T = list(range(1, 16))

    # Shift set: try user's builder or fallback
    if hasattr(opt_mod, "build_shift_set"):
        try:
            S = opt_mod.build_shift_set(T, 4, 8)
        except Exception:
            S = build_shift_set_fallback(T, 4, 8)
    else:
        S = build_shift_set_fallback(T, 4, 8)

    # Min/Max hours
    MinHw = {row["name"]: float(row["min_week_hours"]) for _, row in staff_df.iterrows()}
    MaxHw = {row["name"]: float(row["max_week_hours"]) for _, row in staff_df.iterrows()}

    # Demand as dict of lists indexed by day, optimizer expects Demand[d][t-1]
    Demand = {d: [float(x) for x in demand_df.iloc[d-1, :].tolist()] for d in D}

    # Call the user's function WITHOUT time_limit kw
    fn = getattr(opt_mod, "build_and_solve_shift_model")
    try:
        res = fn(W, D, T, S, MinHw, MaxHw, Demand, Max_Deviation=max_dev)
    except TypeError:
        try:
            res = fn(W, D, T, S, MinHw, MaxHw, Demand, Max_Deviation=max_dev, time_limit=None)
        except Exception as e:
            st.error(f"Failed to call build_and_solve_shift_model without a time limit: {e}")
            st.stop()

    # Normalize to our expected outputs
    out = {}
    out["status"] = res.get("status", "OK")
    out["objective"] = res.get("objective", float("nan"))
    out["elapsed_time"] = res.get("elapsed_time", float("nan"))

    # Convert schedule list[(w,d,t)] to DataFrames
    schedule = res.get("schedule", [])
    # Coverage by day-slot
    rows = []
    for d in D:
        for t in T:
            staffed = sum(1 for (w_, d_, t_) in schedule if d_ == d and t_ == t)
            demand_val = float(demand_df.iloc[d-1, t-1])
            under = max(0.0, demand_val - staffed)
            over = max(0.0, staffed - demand_val)
            rows.append({"day": d, "slot": t, "demand": demand_val, "staffed": staffed, "under": under, "over": over})
    out["coverage_df"] = pd.DataFrame(rows)

    # Hours per worker (each slot counts as 1 hour)
    hours_rows = []
    for w in W:
        total_hours = sum(1 for (w_, _, _) in schedule if w_ == w)
        hours_rows.append({"name": w, "total_hours": total_hours,
                           "min_week_hours": MinHw[w], "max_week_hours": MaxHw[w]})
    out["hours_df"] = pd.DataFrame(hours_rows)

    # Assignments in slot form (each assigned slot as a 1-hour segment)
    out["assignments_df"] = pd.DataFrame([{"name": w, "day": d, "start_slot": t, "end_slot": t, "hours": 1}
                                          for (w, d, t) in schedule])
    return out

def call_any_solver(opt_module, demand_df, staff_df, S, max_deviation):
    """
    Generic fallback call if a different function name is used. No time_limit passed.
    """
    if opt_module is None:
        debug_import_error()
        st.stop()

    candidate_names = ["solve_schedule","schedule","solve","optimize","optimise","run","main"]
    fn = None
    for name in candidate_names:
        if hasattr(opt_module, name) and callable(getattr(opt_module, name)):
            fn = getattr(opt_module, name); break
    if fn is None:
        st.error("No solver function found in optimizer.py. "
                 "Either provide `build_and_solve_shift_model` or one of: "
                 + ", ".join(candidate_names))
        st.stop()

    import inspect as _inspect
    sig = _inspect.signature(fn)
    params = sig.parameters
    kw = {}
    if "demand_df" in params: kw["demand_df"] = demand_df
    elif "demand" in params: kw["demand"] = demand_df
    elif "demand_matrix" in params: kw["demand_matrix"] = demand_df
    elif "sales" in params: kw["sales"] = demand_df

    if "staff_df" in params: kw["staff_df"] = staff_df
    elif "staff" in params: kw["staff"] = staff_df

    if "S" in params: kw["S"] = S
    elif "shifts" in params: kw["shifts"] = S

    if "max_deviation" in params: kw["max_deviation"] = max_deviation
    elif "max_dev" in params: kw["max_dev"] = max_deviation

    res = fn(**kw)

    if isinstance(res, dict):
        return res
    elif isinstance(res, (list, tuple)):
        labels = ["coverage_df","hours_df","assignments_df"]
        out = {}
        for i,obj in enumerate(res):
            key = labels[i] if i < len(labels) else f"out_{i}"
            out[key] = obj
        out.setdefault("status","OK"); out.setdefault("objective", float("nan"))
        return out
    else:
        return {"raw_result": res, "status":"OK", "objective": float("nan")}

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="Shift Scheduler", layout="wide")
st.title("Shift Scheduler (Streamlit + PuLP)")

SLOT_LABELS = ["10-11","11-12","12-13","13-14","14-15","15-16","16-17","17-18","18-19","19-20","20-21","21-22","22-23","23-00","00-01"]
DAY_LABELS = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

with st.sidebar:
    st.header("Configuration")
    max_dev = st.number_input("Max deviation per slot (people)", min_value=0.0, value=2.5, step=0.5)

    st.markdown("---")
    st.subheader("Demand CSV")
    st.caption("Upload a 7x15 CSV (no header): rows=Mon..Sun, cols=15 hourly slots (10-11,...,00-01).")
    demand_file = st.file_uploader("Upload sales_demand_template.csv", type=["csv"])

    st.markdown("---")
    st.subheader("Staff table")
    st.caption("Edit staff below. You can add or delete rows. Download/Upload to reuse.")
    uploaded_staff = st.file_uploader("Upload staff CSV (optional)", type=["csv"], key="staff_csv")
    if "staff_df" not in st.session_state:
        st.session_state["staff_df"] = pd.DataFrame(DEFAULT_STAFF)
    if uploaded_staff is not None:
        st.session_state["staff_df"] = pd.read_csv(uploaded_staff)
    if st.button("Load default staff (10)"):
        st.session_state["staff_df"] = pd.DataFrame(DEFAULT_STAFF)

    staff_df = st.data_editor(
        st.session_state["staff_df"],
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "name": st.column_config.TextColumn("name", help="Unique worker name"),
            "min_week_hours": st.column_config.NumberColumn("min work hour", help="Min weekly hours"),
            "max_week_hours": st.column_config.NumberColumn("max work hour", help="Max weekly hours"),
            "contract_hours": st.column_config.NumberColumn("contract hours", help="Reference contract hours"),
            "weekend_only": st.column_config.CheckboxColumn("weekend only", help="If True, only Fri/Sat/Sun"),
        },
        hide_index=True
    )
    st.session_state["staff_df"] = staff_df
    st.caption(f"Current staff count: **{len(staff_df)}**")

    staff_csv = staff_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download staff CSV", staff_csv, file_name="staff.csv", mime="text/csv")

# Load demand (or demo)
if demand_file is not None:
    demand = pd.read_csv(demand_file, header=None)
else:
    st.info("Using a demo 7x15 demand matrix (no upload).")
    demand = pd.DataFrame(np.array([
        [0.00,0.89,1.08,1.15,2.51,3.11,2.16,4.06,1.64,1.45,1.31,2.68,2.73,2.14,0.86],
        [0.37,1.08,0.90,0.59,2.64,3.40,3.26,3.97,0.86,1.51,1.63,1.77,2.53,2.58,0.07],
        [0.12,0.80,1.67,2.64,2.43,2.64,2.87,2.25,2.61,1.62,1.60,0.88,1.90,2.25,0.72],
        [0.63,1.00,1.67,2.46,1.56,1.91,2.58,2.04,2.63,2.11,1.04,1.34,2.31,2.12,0.61],
        [0.31,0.74,1.39,1.88,2.77,1.75,4.15,3.55,1.85,2.22,1.57,1.34,3.27,3.07,0.76],
        [0.66,0.48,0.64,1.05,1.85,3.61,4.63,3.06,1.99,2.04,1.77,1.82,2.87,3.40,0.88],
        [0.26,0.52,1.46,2.39,1.43,3.18,3.79,3.23,2.91,1.41,2.06,2.28,2.18,2.03,0.86],
    ]))

# Validate shape
if demand.shape != (7,15):
    st.error(f"Demand CSV must be 7 rows x 15 columns. Current shape: {demand.shape}")
    st.stop()

# Build shift set for fallback path (used by generic solver call)
T = list(range(1,16))
if opt_mod is not None and hasattr(opt_mod, "build_shift_set"):
    try:
        S = opt_mod.build_shift_set(T, 4, 8)
    except Exception:
        S = build_shift_set_fallback(T, 4, 8)
else:
    S = build_shift_set_fallback(T, 4, 8)

# --------- Visualization helper (WEEKLY ONLY) ---------

def render_weekly_demand_staffing_chart(coverage_df, SLOT_LABELS):
    st.markdown("### Weekly Demand vs Staffing (average per day)")

    # Average per slot across the 7 days
    avg_df = (coverage_df.groupby("slot")
                           .agg({"demand":"mean", "staffed":"mean"})
                           .reset_index()
                           .sort_values("slot"))

    x_labels = SLOT_LABELS if len(avg_df) == len(SLOT_LABELS) else [str(s) for s in avg_df["slot"]]

    line_df = pd.DataFrame({
        "Demand (avg/day)": avg_df["demand"].to_numpy(),
        "Staffed (avg/day)": avg_df["staffed"].to_numpy(),
    }, index=x_labels)
    st.line_chart(line_df)

    # Metrics: computed on daily values (not averages)
    # Cap applies per day-slot, so we use daily deviations here
    max_abs_daily_dev = float((coverage_df["staffed"] - coverage_df["demand"]).abs().max())
    total_under_week = float((coverage_df["demand"] - coverage_df["staffed"]).clip(lower=0).sum())
    total_over_week  = float((coverage_df["staffed"] - coverage_df["demand"]).clip(lower=0).sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Max |daily deviation| (should ≤ cap)", f"{max_abs_daily_dev:.2f}")
    c2.metric("Total under (week sum)", f"{total_under_week:.1f}")
    c3.metric("Total over (week sum)", f"{total_over_week:.1f}")


# --------- Main action ---------
st.markdown("### Run Optimizer")
if st.button("Solve now", type="primary"):
    with st.spinner("Solving..."):
        # Prefer canonical adapter; else fallback generic caller
        res = adapt_to_user_optimizer(demand, st.session_state["staff_df"], max_dev)
        if res is None:
            res = call_any_solver(opt_mod, demand, st.session_state["staff_df"], S, max_deviation=max_dev)

    st.success(f"Status: {res.get('status','N/A')}, Objective (total deviation): {res.get('objective', float('nan')):.4f}")

    if 'hours_df' in res:
        st.write("Weekly hours per worker")
        st.dataframe(res['hours_df'], use_container_width=True)

    if 'coverage_df' in res:
        st.write("Coverage by day-slot (demand / staffed / under / over)")
        st.dataframe(res['coverage_df'], use_container_width=True)
        # WEEKLY ONLY CHART
        render_weekly_demand_staffing_chart(res["coverage_df"], SLOT_LABELS)

    # Per-worker 7x15 with color highlight
    st.markdown("### Per-worker Schedule (7×15, color = scheduled)")
    assignments_df = res.get("assignments_df", pd.DataFrame(columns=["name","day","start_slot","end_slot"]))
    workers = list(st.session_state["staff_df"]['name'])

    def style_schedule(df):
        return df.style.apply(lambda s: ['background-color: #E6F4FF' if v==1 else '' for v in s], axis=1)

    worker_tables = {}
    import numpy as _np
    if not assignments_df.empty:
        for w in workers:
            mat = _np.zeros((7,15), dtype=int)
            subset = assignments_df[assignments_df['name'] == w]
            for _, row in subset.iterrows():
                d = int(row['day']) - 1
                s = int(row['start_slot']) - 1
                e = int(row['end_slot']) - 1
                s = max(0, min(14, s))
                e = max(0, min(14, e))
                mat[d, s:e+1] = 1
            df = pd.DataFrame(mat, columns=SLOT_LABELS, index=DAY_LABELS)
            worker_tables[w] = df

    for w in workers:
        if w in worker_tables:
            with st.expander(f"{w}"):
                st.dataframe(style_schedule(worker_tables[w]), use_container_width=True)

    # Download per-worker Excel
    if worker_tables:
        try:
            import xlsxwriter
            engine = "xlsxwriter"
        except Exception:
            engine = None
        output = BytesIO()
        with pd.ExcelWriter(output, engine=engine) as writer:
            for w, df in worker_tables.items():
                sheet_name = w[:31] if w else "Worker"
                df.to_excel(writer, sheet_name=sheet_name)
        st.download_button(
            "Download per-worker schedule (Excel)",
            data=output.getvalue(),
            file_name="per_worker_schedule.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
