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

BRAND_GREEN   = "#22c55e"
BRAND_AMBER   = "#f59e0b"
BRAND_PINKRED = "#620332"
BRAND_MUTED   = "#94a3b8"

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
min_count = st.sidebar.number_input("Minimum sample size per group", min_value=1, value=1, step=1)
top_k = st.sidebar.number_input("Show top-K groups", min_value=1, value=25, step=1)

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

def _brandify(fig):
    fig.update_layout(
        template="plotly_white",
        font=dict(size=13),
        title_x=0.01,
        margin=dict(l=10, r=10, t=50, b=60),
        # hoverlabel=dict(bgcolor="white"),
    )
    return fig

def _sorted_bar(
    g: pd.DataFrame,
    y_col: str,
    title: str,
    yaxis_title: str,
    is_percent: bool = False,
    x_col: str = "Category",
    color_seq=None,
):
    # Always grab Category + y_col, and add N if available but not duplicate
    cols = [x_col, y_col]
    if "N" in g.columns and "N" not in cols:
        cols.append("N")

    tmp = g[cols].dropna(subset=[y_col]).copy()

    # Safe numeric coercion if Series, skip if it's already int
    if pd.api.types.is_numeric_dtype(tmp[y_col]):
        pass  # already numeric
    else:
        tmp[y_col] = pd.to_numeric(tmp[y_col], errors="coerce")

    tmp = tmp.sort_values(by=y_col, ascending=False)
    cat_order = tmp[x_col].astype(str).tolist()

    fig = px.bar(
        tmp,
        x=x_col,
        y=y_col,
        title=title,
        color_discrete_sequence=color_seq or [BRAND_BLUE],
        text=tmp["N"] if "N" in tmp.columns else None,
    )
    fig.update_traces(
        textposition="outside",
        texttemplate="N = %{text}" if "N" in tmp.columns else None,
    )

    fig.update_xaxes(categoryorder="array", categoryarray=cat_order, tickangle=-30)
    fig.update_layout(
        xaxis_title="",
        yaxis_title=yaxis_title,
        legend_title="",
        # plot_bgcolor="rgba(0,0,0,0)",
        uniformtext_minsize=10,
        uniformtext_mode="show",
    )

    if is_percent:
        fig.update_yaxes(ticksuffix="%")
        fig.update_traces(hovertemplate="<b>%{x}</b><br>%{y:.1f}%<extra></extra>")
    else:
        if y_col == "N":
            fig.update_traces(hovertemplate="<b>%{x}</b><br>N: %{y:.0f}<extra></extra>")
        else:
            fig.update_traces(hovertemplate="<b>%{x}</b><br>%{y:.2f}<extra></extra>")

    return _brandify(fig)


def samples_bar(g, title_label):
    return _sorted_bar(
        g, "N",
        f"Samples (N) per group — {title_label}",
        "Samples (N)",
        is_percent=False,
        color_seq=[BRAND_MUTED],           # performance-neutral: blue
    )

def early_churn_bar(g, title_label):
    return _sorted_bar(
        g, "Early_Churn_Pct",
        f"Early Churn % per group — {title_label}",
        "% of hires",
        is_percent=True,
        color_seq=[BRAND_PINKRED],         # risk: purple
    )

def ideal_pct_bar(g, title_label):
    return _sorted_bar(
        g, "Ideal_Pct",
        f"Ideal % per group — {title_label}",
        "% of hires",
        is_percent=True,
        color_seq=[BRAND_BLUE],         # ideal profile: purple
    )

def mean_sql_bar(g, title_label):
    return _sorted_bar(
        g, "mean_SQL",
        f"Mean Avg. SQL / Week per group — {title_label}",
        "Avg. SQL / Week",
        is_percent=False,
        color_seq=[BRAND_PURPLE],           # performance: blue
    )

def mean_weeks_bar(g, title_label):
    return _sorted_bar(
        g, "mean_Weeks",
        f"Mean Weeks Active per group — {title_label}",
        "Weeks Active",
        is_percent=False,
        color_seq=[BRAND_YELLOW],         # tenure: yellow
    )

def _box_with_mean_sd(df_source: pd.DataFrame, category_col: str, value_col: str, title: str, brand_color: str):
    # sort categories by group mean desc for stable order
    order = (
        df_source.groupby(category_col)[value_col]
        .mean().sort_values(ascending=False).index.tolist()
    )

    base = df_source[[category_col, value_col]].dropna()
    fig = px.box(
        base, x=category_col, y=value_col, points="outliers",
        color_discrete_sequence=[brand_color],
        title=title,
    )
    fig.update_xaxes(categoryorder="array", categoryarray=order, tickangle=-30)
    fig.update_layout(xaxis_title="", yaxis_title=value_col, legend_title="")
    # means & stds
    stats = base.groupby(category_col)[value_col].agg(["mean", "std"]).reindex(order)
    fig.add_trace(
        go.Scatter(
            x=stats.index,
            y=stats["mean"],
            mode="markers",
            name="mean ± SD",
            marker=dict(color=brand_color, size=8, symbol="diamond"),
            error_y=dict(
                type="data",
                array=stats["std"],
                visible=True,
                thickness=1,
                width=3,
            ),
            customdata=np.stack([stats["mean"], stats["std"]], axis=-1),
            hovertemplate="<b>%{x}</b><br>"
                        "Mean ± SD: %{customdata[0]:.2f} ± %{customdata[1]:.2f}"
        )
    )

    return _brandify(fig)

def boxplot_sql(df_source: pd.DataFrame, category_col: str, title_label: str):
    return _box_with_mean_sd(
        df_source, category_col, "Avg. SQL / Week",
        f"Box Plot — Avg. SQL / Week by {title_label}",
        BRAND_PURPLE
    )

def boxplot_weeks(df_source: pd.DataFrame, category_col: str, title_label: str):
    return _box_with_mean_sd(
        df_source, category_col, "Weeks Active",
        f"Box Plot — Weeks Active by {title_label}",
        BRAND_YELLOW
    )

def bubble_chart(g, title_label):
    g = g.copy()
    g["label"] = g["Category"].astype(str) + " (" + g["N"].astype(str) + ")"
    fig = px.scatter(
        g,
        x="High_SQL_Pct",
        y="Long_Tenure_Pct",
        size="N",
        color="Ideal_Pct",
        text="label",
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
        title=f"Quality vs. Retention — {title_label} (bubble = N, color = Ideal %)",
    )
    fig.update_traces(textposition="top center", textfont=dict(size=12))
    fig.update_layout(
        xaxis_title="High SQL %",
        yaxis_title="Long Tenure %",
        coloraxis_colorbar_title="Ideal %",
        xaxis=dict(rangemode="tozero"),
        yaxis=dict(rangemode="tozero"),
        margin=dict(l=10, r=10, t=50, b=60),
    )
    return fig

def section_pair(df_source, col_label, dfcol): 
    g = screen(df_source, dfcol, min_count=min_count, top_k=top_k) 
    
    if g.empty: 
        st.info(f"Not enough data for {col_label} (increase Top-K / decrease Min Count).") 
        return 
    
    with st.expander(f"{col_label}", expanded=False):
        st.markdown(f"# **{col_label}**") 
        c1, c2 = st.columns(2) 
        with c1: 
            st.plotly_chart(bubble_chart(g, col_label), use_container_width=True)
            st.plotly_chart(mean_sql_bar(g, col_label), use_container_width=True)
            st.plotly_chart(samples_bar(g, col_label), use_container_width=True)
            st.plotly_chart(boxplot_sql(df_source, dfcol, col_label), use_container_width=True)
        with c2: 
            st.plotly_chart(ideal_pct_bar(g, col_label), use_container_width=True)
            st.plotly_chart(mean_weeks_bar(g, col_label), use_container_width=True)
            st.plotly_chart(early_churn_bar(g, col_label), use_container_width=True)
            st.plotly_chart(boxplot_weeks(df_source, dfcol, col_label), use_container_width=True)
            

# 3.1 University Tier
section_pair(df_work, "University Tier", "University Tier")

# 3.2 University Major Bucket
section_pair(df_work, "University Major", "University Major Bucket")

# 3.5 University Graduation Year
section_pair(df_work, "Graduation Year", "University Graduation Year")

# 3.3 High School Tier
section_pair(df_work, "High School Tier", "High School Tier")

# 3.4 Frequent Job Changes
section_pair(df_work, "Frequent Job Changes", "Frequent Job Changes")

# 3.8 Frequent Job Changes Recently
section_pair(df_work, "Recent Job Changes", "Frequent Job Changes Recently")

# 3.6 Most Recent Company
section_pair(df_work, "Most Recent Company", "Most Recent Company")

# 3.7 Most Recent Company Role
section_pair(df_work, "Most Recent Role", "Most Recent Company Role")

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
