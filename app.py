# # Streamlit Data Analytics Playground for Sales Rep Pre-Hire Predictors
# # ---------------------------------------------------------------
# # Features in this MVP:
# # 1) Data loading (CSV/Parquet) + light type coercion
# # 2) Airtable-like dynamic filtering (by text/categorical, numeric, dates, list column "Past Companies")
# # 3) Univariate visuals
# #    - Categorical predictor → aggregated bar of target (mean/median/count)
# #    - Continuous predictor → scatter with trendline
# # 4) Correlation analysis (one-hot encode categoricals, numeric correlations to chosen target)
# # ---------------------------------------------------------------

# import ast
# import io
# from typing import List, Tuple
# from pathlib import Path
# import numpy as np
# import pandas as pd
# import streamlit as st
# import pandas.api.types as ptypes
# import plotly.express as px

# # ------------------------------
# # Config
# # ------------------------------
# st.set_page_config(page_title="Sales Rep Analytics", layout="wide")
# st.title("📊 Sales Rep Analytics Playground — Pre-Hire Predictors")

# # ------------------------------
# # Helpers
# # ------------------------------
# IGNORED_COLS = {
#     "LinkedIn Link", "Lever", "Resume", "Call Recordings",
#     "Current Net Base & OTE (Not Gross) (from Lever Feedback Forms)",
# }

# TARGET_OPTS = ["Avg. SQL / Week", "Weeks Active"]
# DATE_COLS_CAND = ["Start Date", "Termination Date"]
# LIST_COLS = ["Past Companies"]

# AGG_FUNCS = {
#     "mean": np.mean,
#     "median": np.median,
#     "count": "count",
# }

# USER_CATEGORICAL = [
#     "Ramping Formula", "Sales Comp Level Lookup", "Hiring Recommendation",
#     "University Tier", "University Major Bucket", "High School Tier",
#     "Status", "Most Recent Company Role"
# ]
# USER_CONTINUOUS = [
#     "Avg. SQL / Week", "Weeks Active", "Most Recent Company Tenure",
#     "Total Years Outbound Experience", "Total Years Customer Service Experience",
#     "Hiring Rating", "University Graduation Year"
# ]

# @st.cache_data(show_spinner=False)
# def load_dataframe(uploaded_file: io.BytesIO | None) -> pd.DataFrame:
#     """Robust loader with local parquet/csv fallback."""
#     df = None
#     try:
#         if uploaded_file is not None:
#             name = uploaded_file.name.lower()
#             if name.endswith((".parquet", ".pq")):
#                 df = pd.read_parquet(uploaded_file)
#             elif name.endswith(".csv"):
#                 df = pd.read_csv(uploaded_file)
#             else:
#                 df = pd.read_csv(uploaded_file)  # default try as csv
#         else:
#             here = Path.cwd()
#             if (here / "cleaned_dataset.parquet").exists():
#                 df = pd.read_parquet(here / "cleaned_dataset.parquet")
#             elif (here / "cleaned_dataset.csv").exists():
#                 df = pd.read_csv(here / "cleaned_dataset.csv")
#             elif Path("/mnt/data/cleaned_dataset.csv").exists():
#                 df = pd.read_csv("/mnt/data/cleaned_dataset.csv")

#         if df is None:
#             st.error("❌ No dataset found. Upload or place cleaned_dataset.parquet next to this script.")
#             st.stop()
#         return df.copy()
#     except Exception as e:
#         st.error(f"Failed to load data: {e}")
#         st.stop()

# def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.copy()

#     # Parse dates
#     for c in DATE_COLS_CAND:
#         if c in df.columns:
#             df[c] = pd.to_datetime(df[c], errors="coerce")

#     # Ensure numeric for known numeric-like columns if present
#     numeric_like = [
#         "Avg. SQL / Week", "Weeks Active", "Most Recent Company Tenure",
#         "Total Years Outbound Experience", "Total Years Customer Service Experience",
#         "Hiring Rating", "University Graduation Year",
#     ]
#     for c in numeric_like:
#         if c in df.columns:
#             df[c] = pd.to_numeric(df[c], errors="coerce")

#     # Booleans
#     for c in ["Frequent Job Changes", "Frequent Job Changes Recently"]:
#         if c in df.columns:
#             if df[c].dtype == object:
#                 df[c] = df[c].astype(str).str.lower().map({"true": True, "false": False})
#             df[c] = df[c].astype("boolean")

#     # List-like columns (stored as Python-literal strings)
#     for c in LIST_COLS:
#         if c in df.columns:
#             def to_list(x):
#                 # None or NaN
#                 if x is None:
#                     return []
#                 if isinstance(x, float) and pd.isna(x):
#                     return []
                
#                 # Already list/tuple/array
#                 if isinstance(x, (list, tuple, np.ndarray)):
#                     return [str(i).strip() for i in x if str(i).strip()]
                
#                 # Strings
#                 if isinstance(x, str):
#                     s = x.strip()
#                     if not s:
#                         return []
#                     try:
#                         v = ast.literal_eval(s)
#                         if isinstance(v, (list, tuple)):
#                             return [str(i).strip() for i in v if str(i).strip()]
#                     except Exception:
#                         pass
#                     # fallback: comma split
#                     return [t.strip() for t in s.split(",") if t.strip()]
                
#                 # Fallback: everything else → wrap in list
#                 return [str(x)]



#     return df


# from pandas.api import types as ptypes

# def detect_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
#     """Return (categorical_cols, continuous_cols, date_cols)."""

#     date_cols = [c for c in df.columns if ptypes.is_datetime64_any_dtype(df[c])]

#     # Continuous numeric
#     num_cols = [c for c in df.columns if ptypes.is_numeric_dtype(df[c])]

#     # Categorical
#     cat_cols = [c for c in df.columns if (
#         ptypes.is_string_dtype(df[c]) or
#         ptypes.is_categorical_dtype(df[c]) or
#         ptypes.is_bool_dtype(df[c])
#     )]

#     # Low-cardinality numeric also treated as categorical
#     for c in num_cols:
#         if c not in TARGET_OPTS and df[c].nunique(dropna=True) <= 12:
#             if c not in cat_cols:
#                 cat_cols.append(c)

#     # Remove ignored
#     cat_cols = [c for c in cat_cols if c not in IGNORED_COLS and c not in TARGET_OPTS]
#     cont_cols = [c for c in num_cols if c not in IGNORED_COLS]

#     return (cat_cols, cont_cols, date_cols)

# # ------------------------------
# # UI — Data Load & Filters
# # ------------------------------
# with st.sidebar:
#     st.header("⚙️ Data & Filters")
#     upl = st.file_uploader("Upload CSV or Parquet (optional)", type=["csv", "parquet", "pq"])

# raw_df = load_dataframe(upl)
# df = coerce_types(raw_df)

# # Use these if non-empty; else fall back to detect_column_types
# if USER_CATEGORICAL or USER_CONTINUOUS:
#     cat_cols = [c for c in USER_CATEGORICAL if c in df.columns]
#     cont_cols = [c for c in USER_CONTINUOUS if c in df.columns]
#     date_cols = [c for c in df.columns if ptypes.is_datetime64_any_dtype(df[c])]
# else:
#     cat_cols, cont_cols, date_cols = detect_column_types(df)

# # cat_cols, cont_cols, date_cols = detect_column_types(df)

# # Build a compact filter UI similar to Airtable
# st.sidebar.subheader("Filters")
# if "filters" not in st.session_state:
#     st.session_state.filters = []  # each filter: dict

# # Add new filter
# if st.sidebar.button("➕ Add filter"):
#     st.session_state.filters.append({
#         "col": None,
#         "op": None,
#         "val": None,
#         "val2": None,
#     })

# # Operators per dtype
# TEXT_OPS = ["is", "is not", "contains", "not contains", "is any of", "is none of", "is missing", "is not missing"]
# NUM_OPS  = ["=", "!=", ">", ">=", "<", "<=", "between", "is missing", "is not missing"]
# DATE_OPS = ["before", "after", "between", "on", "not on", "is missing", "is not missing"]
# LIST_OPS = ["contains any of", "contains all of", "contains none of", "is missing", "is not missing"]

# # Render existing filters
# for i, f in enumerate(st.session_state.filters):
#     with st.sidebar.expander(f"Filter {i+1}", expanded=False):
#         col = st.selectbox("Column", options=[c for c in df.columns if c not in IGNORED_COLS], index=None, key=f"col_{i}")
#         f["col"] = col
#         if col:
#             series = df[col]
#             # Choose ops
#             if col in LIST_COLS:
#                 op = st.selectbox("Operator", LIST_OPS, index=0, key=f"op_{i}")
#                 f["op"] = op
#                 if op in {"contains any of", "contains all of", "contains none of"}:
#                     multival = st.text_input("Enter comma-separated values", key=f"val_{i}")
#                     f["val"] = [v.strip() for v in multival.split(",") if v.strip()]
#             elif pd.api.types.is_numeric_dtype(series):
#                 op = st.selectbox("Operator", NUM_OPS, index=0, key=f"op_{i}")
#                 f["op"] = op
#                 if op == "between":
#                     v1 = st.number_input("Min", value=float(np.nanmin(series.dropna())) if series.dropna().size else 0.0, key=f"val1_{i}")
#                     v2 = st.number_input("Max", value=float(np.nanmax(series.dropna())) if series.dropna().size else 0.0, key=f"val2_{i}")
#                     f["val"], f["val2"] = v1, v2
#                 elif op not in {"is missing", "is not missing"}:
#                     v = st.number_input("Value", key=f"val_{i}")
#                     f["val"] = v
#             elif ptypes.is_datetime64_any_dtype(series):
#                 op = st.selectbox("Operator", DATE_OPS, index=0, key=f"op_{i}")
#                 f["op"] = op
#                 if op == "between":
#                     d1 = st.date_input("Start date", key=f"val1_{i}")
#                     d2 = st.date_input("End date", key=f"val2_{i}")
#                     f["val"], f["val2"] = pd.to_datetime(d1), pd.to_datetime(d2)
#                 elif op in {"before", "after", "on", "not on"}:
#                     d = st.date_input("Date", key=f"val_{i}")
#                     f["val"] = pd.to_datetime(d)
#             else:  # text/categorical/boolean
#                 op = st.selectbox("Operator", TEXT_OPS, index=0, key=f"op_{i}")
#                 f["op"] = op
#                 if op in {"is", "is not", "contains", "not contains"}:
#                     val = st.text_input("Value", key=f"val_{i}")
#                     f["val"] = val
#                 elif op in {"is any of", "is none of"}:
#                     multival = st.text_input("Enter comma-separated values", key=f"valany_{i}")
#                     f["val"] = [v.strip() for v in multival.split(",") if v.strip()]
#         # Remove filter
#         if st.button("🗑️ Remove", key=f"rm_{i}"):
#             st.session_state.filters.pop(i)
#             st.rerun()


# def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
#     out = df.copy()
#     for f in st.session_state.filters:
#         col, op, v, v2 = f.get("col"), f.get("op"), f.get("val"), f.get("val2")
#         if not col or not op:
#             continue
#         s = out[col]
#         if col in LIST_COLS:
#             if op == "is missing":
#                 out = out[s.apply(lambda x: len(x) == 0)]
#             elif op == "is not missing":
#                 out = out[s.apply(lambda x: len(x) > 0)]
#             elif op == "contains any of":
#                 vals = set(v)
#                 out = out[s.apply(lambda lst: any(item in lst for item in vals))]
#             elif op == "contains all of":
#                 vals = set(v)
#                 out = out[s.apply(lambda lst: vals.issubset(set(lst)))]
#             elif op == "contains none of":
#                 vals = set(v)
#                 out = out[~s.apply(lambda lst: any(item in lst for item in vals))]
#         elif pd.api.types.is_numeric_dtype(s):
#             if op == "is missing":
#                 out = out[s.isna()]
#             elif op == "is not missing":
#                 out = out[s.notna()]
#             elif op == "between":
#                 out = out[(s >= v) & (s <= v2)]
#             elif op == "=":
#                 out = out[s == v]
#             elif op == "!=":
#                 out = out[s != v]
#             elif op == ">":
#                 out = out[s > v]
#             elif op == ">=":
#                 out = out[s >= v]
#             elif op == "<":
#                 out = out[s < v]
#             elif op == "<=":
#                 out = out[s <= v]
#         elif ptypes.is_datetime64_any_dtype(s):
#             if op == "is missing":
#                 out = out[s.isna()]
#             elif op == "is not missing":
#                 out = out[s.notna()]
#             elif op == "before":
#                 out = out[s < v]
#             elif op == "after":
#                 out = out[s > v]
#             elif op == "between":
#                 out = out[(s >= v) & (s <= v2)]
#             elif op == "on":
#                 out = out[s.dt.date == pd.to_datetime(v).date()]
#             elif op == "not on":
#                 out = out[s.dt.date != pd.to_datetime(v).date()]
#         else:
#             ser = s.astype(str)
#             if op == "is missing":
#                 out = out[s.isna() | (ser.str.len() == 0)]
#             elif op == "is not missing":
#                 out = out[s.notna() & (ser.str.len() > 0)]
#             elif op == "is":
#                 out = out[ser == str(v)]
#             elif op == "is not":
#                 out = out[ser != str(v)]
#             elif op == "contains":
#                 out = out[ser.str.contains(str(v), case=False, na=False)]
#             elif op == "not contains":
#                 out = out[~ser.str.contains(str(v), case=False, na=False)]
#             elif op == "is any of":
#                 choices = set(map(str, v))
#                 out = out[ser.isin(choices)]
#             elif op == "is none of":
#                 choices = set(map(str, v))
#                 out = out[~ser.isin(choices)]
#     return out


# filtered_df = apply_filters(df)

# # ------------------------------
# # Section 1 — Univariate Visuals
# # ------------------------------
# st.header("1) Univariate Visuals")

# col_l, col_r = st.columns(2)
# with col_l:
#     target = st.selectbox("Target variable", TARGET_OPTS, index=0)
# with col_r:
#     agg = st.selectbox("Aggregation (for categorical)", list(AGG_FUNCS.keys()), index=0)

# st.subheader("A. Categorical → Aggregated Bar")
# cat_sel = st.selectbox("Categorical predictor", options=sorted([c for c in cat_cols if c not in TARGET_OPTS and c not in IGNORED_COLS]))
# if cat_sel:
#     g = filtered_df[[cat_sel, target]].copy()
#     # Drop NaNs in target for computation
#     if agg in {"mean", "median"}:
#         grouped = g.groupby(cat_sel, dropna=False, observed=True)[target].agg(AGG_FUNCS[agg]).reset_index()
#         grouped = grouped.sort_values(by=target, ascending=False)
#         fig = px.bar(grouped, x=cat_sel, y=target)
#     else:  # count
#         grouped = g.groupby(cat_sel, dropna=False, observed=True)[target].agg("count").reset_index(name="count")
#         grouped = grouped.sort_values(by="count", ascending=False)
#         fig = px.bar(grouped, x=cat_sel, y="count")
#     fig.update_layout(margin=dict(l=10, r=10, t=30, b=60), xaxis_title=cat_sel)
#     st.plotly_chart(fig, use_container_width=True)

# st.subheader("B. Continuous → Scatter with Trendline")
# cont_options = [c for c in cont_cols if c not in TARGET_OPTS]
# cont_sel = st.selectbox("Continuous predictor", options=sorted(cont_options))
# if cont_sel:
#     g2 = filtered_df[[cont_sel, target]].dropna(subset=[cont_sel, target]).copy()
#     if g2.empty:
#         st.info("No data available after filtering for this combination.")
#     else:
#         # Trendline='ols' needs statsmodels; if missing, fallback to no trendline
#         try:
#             fig2 = px.scatter(g2, x=cont_sel, y=target, trendline="ols")
#         except Exception:
#             fig2 = px.scatter(g2, x=cont_sel, y=target)
#         fig2.update_layout(margin=dict(l=10, r=10, t=30, b=60))
#         st.plotly_chart(fig2, use_container_width=True)


# # ------------------------------
# # Section 2 — Correlation Analysis
# # ------------------------------
# st.header("2) Correlation Analysis (categoricals → dummies, booleans → 0/1)")

# method = st.selectbox(
#     "Correlation method", ["pearson", "spearman"],
#     help="Computed on numeric/dummified features vs selected target."
# )

# # 1) Prep dataframe (remove ignored, ensure numeric target)
# work_df = filtered_df.drop(columns=list(IGNORED_COLS & set(filtered_df.columns)), errors="ignore").copy()
# work_df[target] = pd.to_numeric(work_df[target], errors="coerce")

# # If target has no valid numbers after filtering, bail early
# if work_df[target].dropna().empty:
#     st.info("No non-missing values for the selected target after filtering.")
# else:
#     # 2) Booleans → numeric 0/1 (do NOT one-hot them)
#     bool_cols = [c for c in work_df.columns if ptypes.is_bool_dtype(work_df[c]) and c != target]
#     for c in bool_cols:
#         work_df[c] = work_df[c].astype("Int64")  # keeps NA

#     # 3) Numeric features (excluding the target)
#     X_num = work_df.drop(columns=[target], errors="ignore").select_dtypes(include=["number"])

#     # 4) Categorical candidates for dummies (exclude bools/dates/list columns)
#     cat_for_dummies = [
#         c for c in work_df.columns
#         if (c not in X_num.columns)                # not already numeric
#            and (c != target)
#            and (c not in LIST_COLS)
#            and (not ptypes.is_datetime64_any_dtype(work_df[c]))
#            and (ptypes.is_object_dtype(work_df[c]) or ptypes.is_categorical_dtype(work_df[c]))
#     ]

#     # Safe get_dummies: if no categoricals, make an empty DF with matching index
#     if len(cat_for_dummies) > 0:
#         dums = pd.get_dummies(
#             work_df[cat_for_dummies],
#             drop_first=True,
#             dummy_na=True,
#             prefix_sep="="    # unique readable names e.g., "Status=Active"
#         )
#     else:
#         dums = pd.DataFrame(index=work_df.index)

#     # 5) Assemble X and compute corr(feature, y)
#     X = pd.concat([X_num, dums], axis=1)

#     # If X is empty after filtering, bail gracefully
#     if X.shape[1] == 0:
#         st.info("No features available for correlation after current filters.")
#     else:
#         X = X.apply(pd.to_numeric, errors="coerce")
#         y = work_df[target].astype("float")

#         corr_series = X.corrwith(y, method=method).dropna()
#         if corr_series.empty:
#             st.info("No valid correlations could be computed (all-NA or constant features after filtering).")
#         else:
#             corr_abs_sorted = corr_series.abs().sort_values(ascending=False)
#             top_k = st.slider("Show top K features",
#                               min_value=5,
#                               max_value=min(50, len(corr_abs_sorted)),
#                               value=min(15, len(corr_abs_sorted)))
#             top_feats = corr_abs_sorted.head(top_k).index.tolist()

#             st.subheader("Top Correlations vs Target")
#             corr_df = corr_series.loc[top_feats].reset_index()
#             corr_df.columns = ["feature", "corr"]

#             fig3 = px.bar(corr_df, x="feature", y="corr")
#             fig3.update_layout(margin=dict(l=10, r=10, t=30, b=100))
#             fig3.update_xaxes(tickangle=45)
#             st.plotly_chart(fig3, use_container_width=True)

#             with st.expander("Show correlation table"):
#                 st.dataframe(corr_df, use_container_width=True)

# st.caption("Booleans are treated as 0/1; categoricals are one-hot encoded (drop_first + NA) with names like 'Status=Active'.")

# app.py
# Leadership report: SDR profile insights (Streamlit + Plotly)
# Requirements covered:
# - Load cleaned_dataset.parquet
# - Target distributions
# - Thresholds with interactive controls (median/mean/top X%/custom) + “terminated median” option for churn
# - Relationships section: guide + risk chart + bubble chart (3.1–3.11) + heatmap (3.12)
# - Min count and Top-K controls
# - Company brand colors used across charts: blue #00a0f9, yellow #ffd53e, purple #bb48dd
# - Hover shows mean SQL, mean Weeks, N (on bubble)

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# Page config & brand constants
# -----------------------------
st.set_page_config(page_title="SDR Hiring Analytics", layout="wide")
BRAND_BLUE   = "#00a0f9"
BRAND_YELLOW = "#ffd53e"
BRAND_PURPLE = "#bb48dd"

# A gentle 3-point continuous colorscale for bubbles/heatmaps
BRAND_COLORSCALE = [
    [0.0, BRAND_YELLOW],
    [0.5, BRAND_PURPLE],
    [1.0, BRAND_BLUE],
]

# -----------------------------
# Data loading
# -----------------------------
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_parquet("cleaned_dataset.parquet")
    # Keep only specified columns
    columns_to_keep = [
        "Name",
        "Alias",
        "Status",
        "Past Companies",
        "Most Recent Company",
        "Most Recent Company Role",
        "Most Recent Company Tenure",
        "High School Tier",
        "University Tier",
        "University Graduation Year",
        "University Major Bucket",
        "Total Years Outbound Experience",
        "Total Years Customer Service Experience",
        "Frequent Job Changes",
        "Frequent Job Changes Recently",
        "Weeks Active",
        "Avg. SQL / Week",
    ]
    df = df[columns_to_keep].copy()

    # Defensive boolean coercion
    for b in ["Frequent Job Changes", "Frequent Job Changes Recently"]:
        if b in df.columns:
            s = df[b].astype("string").str.strip().str.lower()
            df[b] = s.map({"true": True, "false": False, "yes": True, "no": False, "1": True, "0": False}).fillna(False)

    # Numeric coercions
    for c in [
        "Most Recent Company Tenure",
        "Total Years Outbound Experience",
        "Total Years Customer Service Experience",
        "Weeks Active",
        "Avg. SQL / Week",
        "University Graduation Year",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

df = load_data()
df_of_interest = df.copy()

# -----------------------------
# Helpers: bucketing and screens
# -----------------------------
def bucket_tenure(x):
    if pd.isna(x):
        return "Missing"
    elif x < 1:
        return "<1 yr"
    elif x < 2:
        return "1–2 yrs"
    elif x < 3:
        return "2–3 yrs"
    else:
        return "3+ yrs"

def bucket_outbound(x):
    if pd.isna(x) or x == 0:
        return "0 yrs"
    elif x < 3:
        return "1–2 yrs"
    elif x < 5:
        return "3–4 yrs"
    else:
        return "5+ yrs"

def bucket_service(x):
    if pd.isna(x) or x == 0:
        return "0 yrs"
    elif x < 3:
        return "1–2 yrs"
    elif x < 5:
        return "3–4 yrs"
    else:
        return "5+ yrs"

def screen(df_source, col, min_count=8, top_k=12):
    g = (
        df_source.groupby(col, dropna=False)
        .agg(
            N=("Name", "size") if "Name" in df_source.columns else (col, "size"),
            mean_SQL=("Avg. SQL / Week", "mean"),
            mean_Weeks=("Weeks Active", "mean"),
            High_SQL_Pct=("High_SQL", lambda s: 100 * s.mean()),
            Long_Tenure_Pct=("Long_Tenure", lambda s: 100 * s.mean()),
            Early_Churn_Pct=("Early_Churn", lambda s: 100 * s.mean()),
            Ideal_Pct=("Ideal_Profile", lambda s: 100 * s.mean()),
        )
        .sort_values("N", ascending=False)
    )
    g = g[g["N"] >= min_count].round(
        {
            "mean_SQL": 3,
            "mean_Weeks": 2,
            "High_SQL_Pct": 1,
            "Long_Tenure_Pct": 1,
            "Early_Churn_Pct": 1,
            "Ideal_Pct": 1,
        }
    )
    # Keep label available even if it's the index
    g = g.reset_index().rename(columns={col: "Category"})
    return g.head(top_k)

def screen_two(df_source, col1, col2, min_count=4):
    g = (
        df_source.groupby([col1, col2], dropna=False)
        .agg(
            N=("Name", "size") if "Name" in df_source.columns else (col1, "size"),
            mean_SQL=("Avg. SQL / Week", "mean"),
            mean_Weeks=("Weeks Active", "mean"),
            High_SQL_Pct=("High_SQL", lambda s: 100 * s.mean()),
            Long_Tenure_Pct=("Long_Tenure", lambda s: 100 * s.mean()),
            Early_Churn_Pct=("Early_Churn", lambda s: 100 * s.mean()),
            Ideal_Pct=("Ideal_Profile", lambda s: 100 * s.mean()),
        )
        .reset_index()
        .sort_values("N", ascending=False)
    )
    g = g[g["N"] >= min_count].round(
        {
            "mean_SQL": 3,
            "mean_Weeks": 2,
            "High_SQL_Pct": 1,
            "Long_Tenure_Pct": 1,
            "Early_Churn_Pct": 1,
            "Ideal_Pct": 1,
        }
    )
    return g

# -----------------------------
# Threshold computation helpers
# -----------------------------
def default_thresholds(df_source):
    p75_sql = float(df_source["Avg. SQL / Week"].quantile(0.75))
    p75_weeks = float(df_source["Weeks Active"].quantile(0.75))
    # Early-churn cutoff: terminated median if present, else 6
    if "Status" in df_source.columns:
        term = df_source.loc[df_source["Status"] == "Terminated", "Weeks Active"].dropna()
        early_churn_cut = int(np.median(term)) if len(term) > 0 else 6
    else:
        early_churn_cut = 6
    return p75_sql, p75_weeks, early_churn_cut

P75_SQL_DEFAULT, P75_WEEKS_DEFAULT, EARLY_CHURN_DEFAULT = default_thresholds(df_of_interest)

def compute_threshold(series, method, custom_value=None, tail="high"):
    """
    method: "Median", "Mean", "Top 5%", "Top 10%", "Top 25%", "Top 50%", "Custom"
    tail: "high" for High SQL/Long Tenure (upper tail),
          "low"  for Early Churn (lower tail)
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return None

    method = method.lower()
    if method == "median":
        return float(s.median())
    if method == "mean":
        return float(s.mean())
    if method.startswith("top "):
        # e.g., "Top 25%" -> q = 0.75 for high tail; for churn we flip to lower tail (0.25)
        pct = int(method.split()[1].strip("%"))
        q = 1.0 - (pct / 100.0)
        if tail == "low":
            q = 1.0 - q  # flip to lower tail
        return float(s.quantile(q))
    if method == "custom":
        if custom_value is None:
            return None
        return float(custom_value)
    return None

# -----------------------------
# Sidebar: Threshold controls
# -----------------------------
st.sidebar.header("Adjust Thresholds")

with st.sidebar.expander("How thresholds work", expanded=False):
    st.markdown(
        "- **High SQL**: Anyone at or above this weekly SQL threshold is considered a high performer.\n"
        "- **Long Tenure**: Anyone at or above this weeks-active threshold is considered a long stayer.\n"
        "- **Early Churn**: Anyone at or below this weeks-active cutoff is considered an early exit.\n"
        "_Tip: Defaults use top 25% for High SQL & Long Tenure, and the median weeks of terminated reps for Early Churn (fallback 6)._"
    )

threshold_options = ["Median", "Mean", "Top 5%", "Top 10%", "Top 25%", "Top 50%", "Custom"]

# Defaults pre-set to analysis choices
sql_method = st.sidebar.selectbox("High SQL threshold method", threshold_options, index=4)  # Top 25%
sql_custom = None
if sql_method == "Custom":
    sql_custom = st.sidebar.number_input("Custom High SQL value", min_value=0.0, value=float(P75_SQL_DEFAULT), step=0.1)

weeks_method = st.sidebar.selectbox("Long Tenure threshold method", threshold_options, index=4)  # Top 25%
weeks_custom = None
if weeks_method == "Custom":
    weeks_custom = st.sidebar.number_input("Custom Long Tenure value (weeks)", min_value=0.0, value=float(P75_WEEKS_DEFAULT), step=1.0)

churn_method = st.sidebar.selectbox("Early Churn cutoff method", threshold_options, index=0)  # Median
churn_custom = None
if churn_method == "Custom":
    churn_custom = st.sidebar.number_input("Custom Early Churn cutoff (weeks)", min_value=0.0, value=float(EARLY_CHURN_DEFAULT), step=1.0)

# Compute thresholds
high_sql_threshold = compute_threshold(df_of_interest["Avg. SQL / Week"], sql_method, sql_custom, tail="high")
long_tenure_threshold = compute_threshold(df_of_interest["Weeks Active"], weeks_method, weeks_custom, tail="high")

# Early Churn must be computed from TERMINATED reps only
term_weeks = df_of_interest.loc[
    df_of_interest["Status"].astype(str).str.strip().str.lower() == "terminated",
    "Weeks Active"
].dropna()

if term_weeks.empty:
    # No terminated data available → safe fallback
    early_churn_threshold = EARLY_CHURN_DEFAULT
else:
    if churn_method == "Custom" and churn_custom is not None:
        early_churn_threshold = int(churn_custom)
    elif churn_method == "Mean":
        early_churn_threshold = int(np.floor(term_weeks.mean()))
    elif churn_method == "Median":
        early_churn_threshold = int(np.median(term_weeks))
    else:
        # For Early Churn we want LOWER-tail cutoffs (e.g., “Top 25%” → bottom 25% of weeks)
        early_churn_threshold = compute_threshold(term_weeks, churn_method, churn_custom, tail="low")

# Guard rails if any is None
if high_sql_threshold is None:
    high_sql_threshold = P75_SQL_DEFAULT
if long_tenure_threshold is None:
    long_tenure_threshold = P75_WEEKS_DEFAULT
if early_churn_threshold is None:
    early_churn_threshold = EARLY_CHURN_DEFAULT

st.sidebar.markdown("**Current thresholds:**")
st.sidebar.write(
    {
        "High SQL": round(high_sql_threshold, 3),
        "Long Tenure (weeks)": int(long_tenure_threshold),
        "Early Churn (≤ weeks)": int(early_churn_threshold),
    }
)

# -----------------------------
# Apply thresholds to derive flags
# -----------------------------
df_work = df_of_interest.copy()
df_work["High_SQL"] = df_work["Avg. SQL / Week"] >= high_sql_threshold
df_work["Long_Tenure"] = df_work["Weeks Active"] >= long_tenure_threshold
df_work["Early_Churn"] = df_work["Weeks Active"] <= early_churn_threshold
df_work["Ideal_Profile"] = df_work["High_SQL"] & df_work["Long_Tenure"]

# Buckets for the continuous screens
df_work["Tenure_Bucket"] = df_work["Most Recent Company Tenure"].apply(bucket_tenure)
df_work["Outbound_Bucket"] = df_work["Total Years Outbound Experience"].apply(bucket_outbound)
df_work["Service_Bucket"] = df_work["Total Years Customer Service Experience"].apply(bucket_service)

# -----------------------------
# Global chart controls
# -----------------------------
st.sidebar.header("Chart Controls")
min_count = st.sidebar.number_input("Minimum sample size per group", min_value=1, value=8, step=1)
top_k = st.sidebar.number_input("Show top-K groups", min_value=1, value=12, step=1)

# -----------------------------
# Header
# -----------------------------
st.title("SDR Hiring Analytics - Leadership Report")
st.caption("Explore which backgrounds are linked to higher weekly SQLs and longer tenure. Adjust thresholds to see how the story changes.")

# -----------------------------
# 1) Target distributions
# -----------------------------
st.subheader("1) How are outcomes distributed?")
col1, col2 = st.columns(2)

with col1:
    fig1 = px.histogram(
        df_work,
        x="Weeks Active",
        nbins=20,
        title="Distribution: Weeks Active",
        opacity=0.9,
        color_discrete_sequence=[BRAND_BLUE],
    )
    fig1.add_vline(x=long_tenure_threshold, line_width=2, line_dash="dash", line_color=BRAND_PURPLE)
    fig1.add_vrect(x0=0, x1=early_churn_threshold, line_width=0, fillcolor=BRAND_YELLOW, opacity=0.2)
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.histogram(
        df_work,
        x="Avg. SQL / Week",
        nbins=20,
        title="Distribution: Average SQLs per Week",
        opacity=0.9,
        color_discrete_sequence=[BRAND_PURPLE],
    )
    fig2.add_vline(x=high_sql_threshold, line_width=2, line_dash="dash", line_color=BRAND_BLUE)
    st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# 2) Thresholds (values shown)
# -----------------------------
st.subheader("2) Current thresholds in use")
c1, c2, c3 = st.columns(3)
c1.metric("High SQL (weekly)", value=str(round(high_sql_threshold, 3)))
c2.metric("Long Tenure (weeks)", value=str(int(long_tenure_threshold)))
c3.metric("Early Churn (≤ weeks)", value=str(int(early_churn_threshold)))

# -----------------------------
# 3) Relationships: Guide + Charts
# -----------------------------
st.subheader("3) Relationships: who tends to succeed, who leaves early?")
with st.expander("How to read the numbers", expanded=True):
    st.markdown(
        "- **Count** = how many hires in the group (sample size)\n"
        "- **Average SQLs/week** = productivity\n"
        "- **Average weeks active** = tenure\n"
        "- **High SQL %** = % at/above the High SQL threshold\n"
        "- **Long Tenure %** = % at/above the Long Tenure threshold\n"
        "- **Early Churn %** = % at/below the Early Churn cutoff\n"
        "- **Ideal %** = % who are BOTH High SQL and Long Tenure"
    )

def risk_chart(g, title_label):
    # long-form for side-by-side bars of Early Churn vs Ideal
    lf = g[["Category", "Early_Churn_Pct", "Ideal_Pct"]].melt(
        id_vars="Category", var_name="Metric", value_name="Percent"
    )
    color_map = {"Early_Churn_Pct": BRAND_PURPLE, "Ideal_Pct": BRAND_BLUE}
    fig = px.bar(
        lf,
        x="Category",
        y="Percent",
        color="Metric",
        color_discrete_map=color_map,
        barmode="group",
        title=f"Risk vs. Win: {title_label}",
    )
    fig.update_layout(xaxis_title="", yaxis_title="% of hires", legend_title="")
    fig.update_traces(hovertemplate="<b>%{x}</b><br>%{legendgroup}: %{y:.1f}%<extra></extra>")
    return fig

def bubble_chart(g, title_label):
    fig = px.scatter(
        g,
        x="High_SQL_Pct",
        y="Long_Tenure_Pct",
        size="N",
        color="Ideal_Pct",
        color_continuous_scale=BRAND_COLORSCALE,
        hover_data={
            "Category": True,
            "mean_SQL": True,
            "mean_Weeks": True,
            "N": True,
            "High_SQL_Pct": True,
            "Long_Tenure_Pct": True,
            "Ideal_Pct": True,
        },
        title=f"Quality vs. Retention: {title_label} (bubble size = Count, color = Ideal %)",
    )
    fig.update_layout(xaxis_title="High SQL %", yaxis_title="Long Tenure %", coloraxis_colorbar_title="Ideal %")
    return fig

def section_pair(df_source, col_label, dfcol):
    g = screen(df_source, dfcol, min_count=min_count, top_k=top_k)
    if g.empty:
        st.info(f"Not enough data for {col_label} (increase Top-K / decrease Min Count).")
        return
    st.markdown(f"**{col_label}**")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(risk_chart(g, col_label), use_container_width=True)
    with c2:
        st.plotly_chart(bubble_chart(g, col_label), use_container_width=True)

# 3.1 University Tier
section_pair(df_work, "University Tier", "University Tier")

# 3.2 University Major Bucket
section_pair(df_work, "University Major", "University Major Bucket")

# 3.3 High School Tier
section_pair(df_work, "High School Tier", "High School Tier")

# 3.4 Frequent Job Changes
section_pair(df_work, "Frequent Job Changes", "Frequent Job Changes")

# 3.5 University Graduation Year
section_pair(df_work, "Graduation Year", "University Graduation Year")

# 3.6 Most Recent Company
section_pair(df_work, "Most Recent Company", "Most Recent Company")

# 3.7 Most Recent Company Role
section_pair(df_work, "Most Recent Role", "Most Recent Company Role")

# 3.8 Frequent Job Changes Recently
section_pair(df_work, "Recent Job Changes", "Frequent Job Changes Recently")

# 3.9 Most Recent Company Tenure (bucketed)
section_pair(df_work, "Tenure at Last Company", "Tenure_Bucket")

# 3.10 Total Years Outbound Experience (bucketed)
section_pair(df_work, "Outbound Experience (Years)", "Outbound_Bucket")

# 3.11 Total Years Customer Service Experience (bucketed)
section_pair(df_work, "Customer Service Experience (Years)", "Service_Bucket")

# 3.12 Recent Company × Tenure Bucket (heatmap)
st.markdown("**Most Recent Company × Tenure at Last Company (Heatmap)**")
g_two = screen_two(df_work, "Most Recent Company", "Tenure_Bucket", min_count=max(1, min_count // 2))
if g_two.empty:
    st.info("Not enough data to draw the heatmap (try lowering the minimum sample size).")
else:
    # Pivot for heatmap (Ideal %)
    pivot = g_two.pivot(index="Most Recent Company", columns="Tenure_Bucket", values="Ideal_Pct")
    # Sort rows by overall Ideal% to make the heatmap more informative
    row_order = pivot.mean(axis=1).sort_values(ascending=False).index
    pivot = pivot.loc[row_order]
    fig_hm = px.imshow(
        pivot,
        text_auto=".1f",
        color_continuous_scale=BRAND_COLORSCALE,
        aspect="auto",
        labels=dict(color="Ideal %"),
        title="Ideal % by Company and Tenure Bucket",
    )
    st.plotly_chart(fig_hm, use_container_width=True)

# -----------------------------
# Footer
# -----------------------------
st.caption(
    "Notes: • All charts and tables update when you change the cutoffs above. "
    "• High SQL = reps at/above the SQL/week cutoff. • Long Tenure = reps at/above the Weeks Active cutoff. "
    "• Early Churn = reps at/below the Early Churn cutoff (computed from TERMINATED reps only). "
    "• % values are within-group shares; N is the number of hires in that group. "
    "• Colors used: performance (blue), tenure (yellow), ideal profile (purple). "
    "• This report uses pre-hire information for analysis; post-hire fields like Status are only used to set the Early Churn cutoff."
)
