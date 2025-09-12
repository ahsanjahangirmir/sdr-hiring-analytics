# pages/02_Correlations.py
from ast import literal_eval
from pathlib import Path
import numpy as np
from collections.abc import Iterable
import pandas as pd
import plotly.graph_objs as go
import streamlit as st

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Correlation Explorer", layout="wide")
st.title("Correlation Explorer")

BRAND_BLUE   = "#00a0f9"
BRAND_YELLOW = "#ffd53e"
BRAND_PURPLE = "#bb48dd"

# -----------------------------
# Load data
# -----------------------------
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    path = Path("cleaned_dataset.parquet")
    if not path.exists():
        st.error("cleaned_dataset.parquet not found in the app directory.")
        st.stop()
    df = pd.read_parquet(path)
    cols_keep = [
        "Name","Alias","Status","Past Companies",
        "Most Recent Company","Most Recent Company Role","Most Recent Company Tenure",
        "High School Tier","University Tier","University Graduation Year","University Major Bucket",
        "Total Years Outbound Experience","Total Years Customer Service Experience",
        "Frequent Job Changes","Frequent Job Changes Recently",
        "Weeks Active","Avg. SQL / Week", "Overall Speaking Score","Composite Speaking Score", "Grammar Score", "Grammar Accurate Sentences (%)", "Good Word Pronunciation (%)", "Fair Word Pronunciation (%)", "Poor Word Pronunciation (%)", 'Talent Signal', 'CCAT Raw Score', 'CCAT Percentile',
       'CCAT Invalid', 'Game Percentile', 'EPP Percent Match', 'EPP Invalid',
       'SalesAP Recommendation', 'SalesAP Invalid', 'SalesAP Recommendation Score'
    ]
    df = df[cols_keep].copy()

    # Coerce continuous metrics to numeric
    for c in [
        "Avg. SQL / Week","Weeks Active",
        "Total Years Outbound Experience","Total Years Customer Service Experience",
        "Most Recent Company Tenure","University Graduation Year", "Overall Speaking Score","Composite Speaking Score", "Grammar Score", "Grammar Accurate Sentences (%)", "Good Word Pronunciation (%)", "Fair Word Pronunciation (%)", "Poor Word Pronunciation (%)", 'Talent Signal', 'CCAT Raw Score', 'CCAT Percentile',
       'CCAT Invalid', 'Game Percentile', 'EPP Percent Match', 'EPP Invalid',
       'SalesAP Recommendation', 'SalesAP Invalid', 'SalesAP Recommendation Score'
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Boolean defensiveness
    for b in ["Frequent Job Changes","Frequent Job Changes Recently"]:
        if b in df.columns:
            s = df[b].astype("string").str.strip().str.lower()
            df[b] = s.map({"true": True,"false": False,"yes": True,"no": False,"1": True,"0": False})

    # Buckets needed for filtering
    def bucket_tenure(x):
        if pd.isna(x): return "Missing"
        if x < 1:  return "<1 yr"
        if x < 2:  return "1–2 yrs"
        if x < 3:  return "2–3 yrs"
        return "3+ yrs"
    df["Tenure_Bucket"] = df["Most Recent Company Tenure"].apply(bucket_tenure)

    def bucket_outbound(x):
        if pd.isna(x) or x == 0: return "0 yrs"
        if x < 3:  return "1–2 yrs"
        if x < 5:  return "3–4 yrs"
        return "5+ yrs"
    df["Outbound_Bucket"] = df["Total Years Outbound Experience"].apply(bucket_outbound)

    def bucket_service(x):
        if pd.isna(x) or x == 0: return "0 yrs"
        if x < 3:  return "1–2 yrs"
        if x < 5:  return "3–4 yrs"
        return "5+ yrs"
    df["Service_Bucket"] = df["Total Years Customer Service Experience"].apply(bucket_service)

    return df

df = load_data()

# -----------------------------
# Helpers
# -----------------------------
CONTINUOUS_CHOICES = [
    "Avg. SQL / Week",
    "Weeks Active",
    "Total Years Outbound Experience",
    "Total Years Customer Service Experience",
    "Most Recent Company Tenure",
    "Overall Speaking Score",
    "Composite Speaking Score",
    "Grammar Score", "Grammar Accurate Sentences (%)", "Good Word Pronunciation (%)", "Fair Word Pronunciation (%)", "Poor Word Pronunciation (%)", 'Talent Signal', 'CCAT Raw Score', 'CCAT Percentile',
       'Game Percentile', 'EPP Percent Match', 'SalesAP Recommendation Score'
    #    'EPP Invalid','CCAT Invalid', 
    #    'SalesAP Recommendation', 'SalesAP Invalid'
]

def _parse_past_companies(val) -> list[str]:
    """Return a clean list of company names from a cell."""
    if val is None:
        return []

    # Case 1: Already a list or array
    if isinstance(val, (list, np.ndarray)):
        seq = list(val)

    # Case 2: String
    elif isinstance(val, str):
        s = val.strip()
        if not s:
            return []
        try:
            parsed = literal_eval(s)
            if isinstance(parsed, (list, tuple, np.ndarray)):
                seq = list(parsed)
            else:
                seq = [s]
        except Exception:
            # Fallback if it's not a literal list string
            # Try splitting on commas
            if "," in s:
                seq = [p.strip() for p in s.split(",")]
            else:
                seq = [s]

    # Case 3: Anything else (numbers, NaN)
    else:
        return []

    # Clean up entries
    out = []
    for x in seq:
        if x is None:
            continue
        s = str(x).strip()
        if s and s.lower() != "nan":
            out.append(s)
    return out

def extract_company_tokens(df: pd.DataFrame, max_tokens: int = 500) -> list[str]:
    """Build a list of unique company names from Past Companies + Most Recent Company."""
    tokens = set()

    if "Past Companies" in df.columns:
        for row in df["Past Companies"].dropna().tolist():
            for tok in _parse_past_companies(row):
                tokens.add(tok)

    if "Most Recent Company" in df.columns:
        for tok in df["Most Recent Company"].dropna().astype(str).str.strip():
            if tok and tok.lower() != "nan":
                tokens.add(tok)

    return sorted(tokens)[:max_tokens]


COMPANY_TOKENS = extract_company_tokens(df)

def apply_filters(data: pd.DataFrame) -> pd.DataFrame:
    """Apply all sidebar filters and return filtered dataframe."""
    filt = data.copy()

    # --- Company filters ---
    with st.sidebar.expander("Filter by Companies", expanded=False):
        # Past Companies include (list membership; any-of)
        pci = st.multiselect(
            "Include anyone who has worked at any of these companies",
            options=COMPANY_TOKENS,
        )
        if pci:
            want = {str(x).strip().lower() for x in pci}
            def _row_matches(val) -> bool:
                have = {s.lower() for s in _parse_past_companies(val)}
                return len(want & have) > 0
            filt = filt[filt["Past Companies"].apply(_row_matches)]

        # Most Recent Company explicit filter
        mrc_sel = st.multiselect(
            "Most Recent Company (exact match)",
            options=sorted(filt["Most Recent Company"].dropna().astype(str).unique().tolist())
        )
        if mrc_sel:
            filt = filt[filt["Most Recent Company"].astype(str).isin(mrc_sel)]

        # Most Recent Company Role filter
        mrr_sel = st.multiselect(
            "Most Recent Company Role (exact match)",
            options=sorted(filt["Most Recent Company Role"].dropna().astype(str).unique().tolist())
        )
        if mrr_sel:
            filt = filt[filt["Most Recent Company Role"].astype(str).isin(mrr_sel)]

    # --- Experience & Tenure buckets ---
    with st.sidebar.expander("Filter by Experience & Tenure Buckets", expanded=False):
        tb = st.multiselect(
            "Recent Company Tenure (bucket)",
            options=["<1 yr","1–2 yrs","2–3 yrs","3+ yrs","Missing"]
        )
        if tb:
            filt = filt[filt["Tenure_Bucket"].isin(tb)]

        ob = st.multiselect(
            "Outbound Experience (bucket)",
            options=["0 yrs","1–2 yrs","3–4 yrs","5+ yrs"]
        )
        if ob:
            filt = filt[filt["Outbound_Bucket"].isin(ob)]

        sb = st.multiselect(
            "Customer Service Experience (bucket)",
            options=["0 yrs","1–2 yrs","3–4 yrs","5+ yrs"]
        )
        if sb:
            filt = filt[filt["Service_Bucket"].isin(sb)]

    # --- Education filters ---
    with st.sidebar.expander("Filter by Education", expanded=False):
        hs = st.multiselect(
            "High School Tier",
            options=sorted(filt["High School Tier"].dropna().astype(str).unique().tolist())
        )
        if hs:
            filt = filt[filt["High School Tier"].astype(str).isin(hs)]

        ut = st.multiselect(
            "University Tier",
            options=sorted(filt["University Tier"].dropna().astype(str).unique().tolist())
        )
        if ut:
            filt = filt[filt["University Tier"].astype(str).isin(ut)]

        um = st.multiselect(
            "University Major Bucket",
            options=sorted(filt["University Major Bucket"].dropna().astype(str).unique().tolist())
        )
        if um:
            filt = filt[filt["University Major Bucket"].astype(str).isin(um)]

        # Grad year range (opt-in; don't drop Unknowns by default)
        if filt["University Graduation Year"].notna().any():
            lo = int(np.nanmin(filt["University Graduation Year"].values))
            hi = int(np.nanmax(filt["University Graduation Year"].values))
            apply_gy = st.checkbox("Filter by graduation year", value=False)
            gy = st.slider("University Graduation Year range", min_value=lo, max_value=hi, value=(lo, hi), disabled=not apply_gy)
            include_unknown = st.checkbox("Include 'Unknown' grad year", value=True, disabled=not apply_gy)
            if apply_gy:
                cond = filt["University Graduation Year"].between(gy[0], gy[1])
                if include_unknown:
                    cond = cond | filt["University Graduation Year"].isna()
                filt = filt[cond]

    # --- Behavior flags ---
    with st.sidebar.expander("Filter by Job-Change Flags", expanded=False):
        fjc = st.selectbox("Frequent Job Changes", options=["All","True","False"], index=0)
        if fjc != "All":
            filt = filt[filt["Frequent Job Changes"] == (fjc == "True")]

        fjcr = st.selectbox("Frequent Job Changes Recently", options=["All","True","False"], index=0)
        if fjcr != "All":
            filt = filt[filt["Frequent Job Changes Recently"] == (fjcr == "True")]

    return filt

# -----------------------------
# UI: Pick variables and filters
# -----------------------------
left, right = st.columns([1,1])

with left:
    x_var = st.selectbox("Select X-axis (continuous variable)", options=CONTINUOUS_CHOICES, index=0)
with right:
    y_var = st.selectbox("Select Y-axis (continuous variable)", options=CONTINUOUS_CHOICES, index=1)

st.sidebar.header("Filters")
filtered = apply_filters(df)

# -----------------------------
# Compute correlations on filtered data
# -----------------------------
# Include Status so we can color points when toggled on
pair_cols = [x_var, y_var] + (["Status"] if "Status" in filtered.columns else [])
pair = filtered[pair_cols].copy()
pair[[x_var, y_var]] = pair[[x_var, y_var]].apply(pd.to_numeric, errors="coerce")
pair = pair.dropna(subset=[x_var, y_var])

# Toggle to color-code by Status
show_status = st.checkbox("Show Status", value=False)

if pair.empty:
    st.warning("No rows left after filtering (or selected columns are empty). Adjust filters or choose different variables.")
    st.stop()

pearson = pair[x_var].corr(pair[y_var], method="pearson")
spearman = pair[x_var].corr(pair[y_var], method="spearman")

def corr_strength(r: float) -> tuple[str, str]:
    a = abs(r)
    if a < 0.10: return ("Negligible", "#94a3b8")
    if a < 0.30: return ("Weak", "#f59e0b")
    if a < 0.50: return ("Moderate", BRAND_BLUE)
    if a < 0.70: return ("Strong", "#22c55e")
    return ("Very strong", BRAND_PURPLE)

ptype, pcolor = corr_strength(pearson)
stype, scolor = corr_strength(spearman)

st.markdown(
    f"""
<div style="display:flex;gap:8px;flex-wrap:wrap;margin:8px 0">
  <div style="padding:6px 10px;border:1px solid #d0d4da;border-radius:999px">
    <b>Rows</b> = {len(pair)}
  </div>
  <div style="padding:6px 10px;border:1px solid #d0d4da;border-radius:999px;background:{pcolor}20">
    Pearson r = {pearson:.4f} · <b style="color:{pcolor}">{ptype}</b>
  </div>
  <div style="padding:6px 10px;border:1px solid #d0d4da;border-radius:999px;background:{scolor}20">
    Spearman ρ = {spearman:.4f} · <b style="color:{scolor}">{stype}</b>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Plotly scatter + best-fit line (with Name/Alias in hover)
# -----------------------------
# Build a dynamic hovertemplate that includes Name/Alias (if present) and Status (if toggled)
hover_cols = [c for c in ["Name", "Alias"] if c in filtered.columns]

def build_hovertemplate(status_label: str | None = None) -> str:
    tpl = f"{x_var}: %{{x:.4f}}<br>{y_var}: %{{y:.4f}}"
    for i, col in enumerate(hover_cols):
        label = "Name" if col == "Name" else "Alias"
        tpl += f"<br>{label}: %{{customdata[{i}]}}"
    if status_label is not None:
        tpl += f"<br>Status: {status_label}"
    return tpl + "<extra></extra>"

def custom_for(subdf: pd.DataFrame):
    return filtered.loc[subdf.index, hover_cols].to_numpy() if hover_cols else None

x = pair[x_var].values
y = pair[y_var].values

# Best-fit line (visual guide only)
show_line = False
try:
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 120)
    y_line = slope * x_line + intercept
    show_line = True
except Exception:
    show_line = False

fig = go.Figure()

if show_status and "Status" in pair.columns:
    st_norm = pair["Status"].astype(str).str.strip().str.lower()
    active = pair[st_norm.eq("active")]
    term   = pair[st_norm.eq("terminated")]
    other  = pair[~(st_norm.eq("active") | st_norm.eq("terminated"))]

    if not active.empty:
        fig.add_trace(
            go.Scattergl(
                x=active[x_var], y=active[y_var],
                mode="markers",
                marker=dict(size=8, color="#2ecc71", line=dict(width=0)),  # green
                name="Active",
                customdata=custom_for(active),
                hovertemplate=build_hovertemplate("Active"),
            )
        )
    if not term.empty:
        fig.add_trace(
            go.Scattergl(
                x=term[x_var], y=term[y_var],
                mode="markers",
                marker=dict(size=8, color="#e74c3c", line=dict(width=0)),  # red
                name="Terminated",
                customdata=custom_for(term),
                hovertemplate=build_hovertemplate("Terminated"),
            )
        )
    if not other.empty:
        fig.add_trace(
            go.Scattergl(
                x=other[x_var], y=other[y_var],
                mode="markers",
                marker=dict(size=8, color="#94a3b8", line=dict(width=0)),  # gray
                name="Other/Unknown",
                customdata=custom_for(other),
                hovertemplate=build_hovertemplate("Other/Unknown"),
            )
        )
else:
    # Single-color scatter (original behavior)
    fig.add_trace(
        go.Scattergl(
            x=x, y=y,
            mode="markers",
            marker=dict(size=8, color=BRAND_BLUE, line=dict(width=0)),
            name="Data points",
            customdata=custom_for(pair),
            hovertemplate=build_hovertemplate(None),
        )
    )

if show_line:
    fig.add_trace(
        go.Scatter(
            x=x_line, y=y_line,
            mode="lines",
            line=dict(width=3, color=BRAND_YELLOW),
            name="Best-fit line",
            hoverinfo="skip",
        )
    )

fig.update_layout(
    xaxis_title=x_var,
    yaxis_title=y_var,
    height=650,
    margin=dict(l=40, r=20, t=30, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

st.plotly_chart(fig, use_container_width=True)


st.caption(
    "Tip: Use the sidebar to narrow the view by company history, education, tenure buckets, and job-change flags. "
    "Correlations recompute on the filtered data only. Blue dots show individuals (or red/green by status when toggled on); "
    "the yellow line is a simple best-fit guide."
)
