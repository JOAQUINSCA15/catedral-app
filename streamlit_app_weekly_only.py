import streamlit as st
import pathlib as pl
import os, sys, math, re
import numpy as np
import pandas as pd
from io import BytesIO

# ------------------ Paths ------------------
ROOT = pl.Path(__file__).parent          # repo folder at runtime
DATA = ROOT / "data"                      # put small input files here in your repo

# ------------------ Page config ------------------
st.set_page_config(page_title="Amorino Sevilla – Weekly Scheduler", layout="wide")
st.title("Amorino Sevilla – Weekly Scheduler")

# ------------------ Helpers ------------------
def _assert_exists(p: pl.Path):
    if not p.exists():
        st.error(f"❌ Missing file: `{p}`.\n\n"
                 "Tip: commit the file to your repo and load it with a **relative** path, "
                 "not a Windows path like `C:\\...`.")
        st.stop()
    return p

@st.cache_data(show_spinner="Loading CSV...")
def load_csv(rel_path: str) -> pd.DataFrame:
    p = _assert_exists(DATA / rel_path)
    return pd.read_csv(p)

@st.cache_data(show_spinner="Loading Excel...")
def load_excel(rel_path: str, sheet_name=0) -> pd.DataFrame:
    p = _assert_exists(DATA / rel_path)
    return pd.read_excel(p, sheet_name=sheet_name)

@st.cache_resource(show_spinner="Initializing model/resources...")
def init_model(*args, **kwargs):
    # TODO: move heavy model loads here (was it happening at import time?)
    # return your_model
    return None

# === Robust upload reader for CSV/Excel with encoding & delimiter handling ===
def read_table_resilient(
    uploaded_file,
    csv_sep=";",
    csv_kwargs=None,
    excel_kwargs=None,
):
    """
    Reads an uploaded file object (Streamlit UploadedFile) as CSV or Excel.
    - Sniffs Excel by magic bytes or extension.
    - For CSV, tries common encodings and uses a fixed delimiter (default ';').
    Returns (df, meta_str) or raises an Exception with a helpful message.
    """
    import io

    if uploaded_file is None:
        raise ValueError("No file provided")

    csv_kwargs = csv_kwargs or {}
    excel_kwargs = excel_kwargs or {}

    # Read raw bytes once
    raw = uploaded_file.read()
    name = getattr(uploaded_file, "name", "").lower()

    # Excel sniff
    is_xlsx = raw[:2] == b"PK"  # ZIP container
    is_xls  = raw[:8] == b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1"  # BIFF

    if name.endswith((".xls", ".xlsx")) or is_xlsx or is_xls:
        # Excel path
        try:
            if name.endswith(".xls") or is_xls:
                df = pd.read_excel(io.BytesIO(raw), engine="xlrd", **excel_kwargs)
                return df, "excel(.xls)"
            else:
                df = pd.read_excel(io.BytesIO(raw), engine="openpyxl", **excel_kwargs)
                return df, "excel(.xlsx)"
        except Exception as e:
            raise RuntimeError(f"Excel parse failed: {e}") from e

    # CSV path
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    last_err = None
    for enc in encodings:
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=enc, sep=csv_sep, **csv_kwargs)
            return df, f"csv({enc})"
        except Exception as e:
            last_err = e
    # Final fallback: replace undecodable bytes
    try:
        text = raw.decode("latin-1", errors="replace")
        df = pd.read_csv(io.StringIO(text), sep=csv_sep, **csv_kwargs)
        return df, "csv(latin-1 with replacement)"
    except Exception as e:
        raise RuntimeError(
            f"CSV parse failed. Last error: {last_err}\nFallback error: {e}"
        ) from e

def normalize_demand_to_7x15(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Make whatever the user uploaded behave like a 7x15 numeric matrix:
    - If it's 7x1 with delimited strings, split by ; , tab or |.
    - If it's 15x7, transpose it.
    - Strip empties, coerce numeric, and clip to 7 rows x 15 cols.
    """
    df = df_in.copy()

    # If everything landed in a single column, try to split
    if df.shape[1] == 1:
        s = df.iloc[:, 0].astype(str).str.strip()
        sep_candidates = [';', '\t', ',', '|']
        best_sep = None
        best_hits = -1
        for sep in sep_candidates:
            # approximate separator count per line (expect ~14 for 15 values)
            hits = s.str.count(re.escape(sep)).median(skipna=True)
            if hits > best_hits:
                best_hits = hits
                best_sep = sep
        if best_sep is not None and best_hits >= 5:  # permissive threshold
            df = s.str.split(best_sep, expand=True)

    # Clean empty strings -> NaN
    df = df.replace(r'^\s*$', np.nan, regex=True)

    # Coerce to numeric
    df = df.apply(pd.to_numeric, errors="coerce")

    # If it's 15x7, transpose to 7x15
    if df.shape == (15, 7):
        df = df.T

    # If it's larger, crop top-left 7x15
    if df.shape[0] >= 7 and df.shape[1] >= 15:
        df = df.iloc[:7, :15]

    return df

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

# ------------------ Labels ------------------
SLOT_LABELS = ["10-11","11-12","12-13","13-14","14-15","15-16","16-17","17-18","18-19","19-20","20-21","21-22","22-23","23-00","00-01"]
DAY_LABELS = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

# ------------------ Sidebar UI ------------------
with st.sidebar:
    st.header("Configuration")
    max_dev = st.number_input("Max deviation per slot (people)", min_value=0.0, value=2.5, step=0.5)

    st.markdown("---")
    st.subheader("Demand file")
    st.caption("Upload a 7×15 CSV/Excel (no header): rows=Mon..Sun, cols=15 hourly slots (10-11,...,00-01).")
    demand_file = st.file_uploader("Upload sales_demand_template (CSV or Excel)", type=["csv", "xls", "xlsx"])

    st.markdown("---")
    st.subheader("Staff table")
    st.caption("Edit staff below. You can add or delete rows. Download/Upload to reuse.")
    uploaded_staff = st.file_uploader("Upload staff CSV/Excel (optional)", type=["csv", "xls", "xlsx"], key="staff_csv")

    if "staff_df" not in st.session_state:
        st.session_state["staff_df"] = pd.DataFrame(DEFAULT_STAFF)

    if uploaded_staff is not None:
        try:
            df_staff, meta = read_table_resilient(
                uploaded_staff,
                csv_sep=";",                 # adjust to "," if your CSVs are comma-separated
                csv_kwargs={"on_bad_lines": "skip"},
                excel_kwargs={},             # e.g., {"sheet_name": 0}
            )
            # Normalize possible alt headers
            df_staff = df_staff.rename(columns={
                "min work hour": "min_week_hours",
                "max work hour": "max_week_hours",
                "contract hours": "contract_hours",
                "weekend only": "weekend_only",
            })
            st.session_state["staff_df"] = df_staff
            st.success(f"Loaded staff from {meta}.")
        except Exception as e:
            st.error(f"Could not read staff file: {e}")

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

# ------------------ Demand load (or demo) ------------------
if demand_file is not None:
    try:
        # Let pandas sniff delimiter for CSV; Excel handled automatically.
        df_demand, meta = read_table_resilient(
            demand_file,
            csv_sep=None,                               # let pandas sniff
            csv_kwargs={"header": None, "engine": "python"},
            excel_kwargs={"header": None, "sheet_name": 0},
        )
        demand = normalize_demand_to_7x15(df_demand)

        # Final sanity: must be 7 x 15
        if demand.shape != (7, 15):
            raise ValueError(f"Demand file must be 7 rows × 15 columns after normalization, got {demand.shape}")

        demand = demand.fillna(0.0)
        st.success(f"Loaded demand from {meta}. Interpreted shape: {demand.shape}")
    except Exception as e:
        st.error(f"Could not read demand file: {e}")
        st.stop()
else:
    st.info("Using a demo 7×15 demand matrix (no upload).")
    demand = pd.DataFrame(np.array([
        [0.00,0.89,1.08,1.15,2.51,3.11,2.16,4.06,1.64,1.45,1.31,2.68,2.73,2.14,0.86],
        [0.37,1.08,0.90,0.59,2.64,3.40,3.26,3.97,0.86,1.51,1.63,1.77,2.53,2.58,0.07],
        [0.12,0.80,1.67,2.64,2.43,2.64,2.87,2.25,2.61,1.62,1.60,0.88,1.90,2.25,0.72],
        [0.63,1.00,1.67,2.46,1.56,1.91,2.58,2.04,2.63,2.11,1.04,1.34,2.31,2.12,0.61],
        [0.31,0.74,1.39,1.88,2.77,1.75,4.15,3.55,1.85,2.22,1.57,1.34,3.27,3.07,0.76],
        [0.66,0.48,0.64,1.05,1.85,3.61,4.63,3.06,1.99,2.04,1.77,1.82,2.87,3.40,0.88],
        [0.26,0.52,1.46,2.39,1.43,3.18,3.79,3.23,2.91,1.41,2.06,2.28,2.18,2.03,0.86],
    ]))

# ------------------ Validate shape ------------------
if demand.shape != (7,15):
    st.error(f"Demand file must be 7 rows × 15 columns. Current shape: {demand.shape}")
    st.stop()

# ------------------ Build shift set for fallback path ------------------
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
    max_abs_daily_dev = float((coverage_df["staffed"] - coverage_df["demand"]).abs().max())
    total_under_week = float((coverage_df["demand"] - coverage_df["staffed"]).clip(lower=0).sum())
    total_over_week  = float((coverage_df["staffed"] - coverage_df["demand"]).clip(lower=0).sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Max |daily deviation| (should ≤ cap)", f"{max_abs_daily_dev:.2f}")
    c2.metric("Total under (week sum)", f"{total_under_week:.1f}")
    c3.metric("Total over (week sum)", f"{total_over_week:.1f}")

# ------------------ Main action ------------------
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
