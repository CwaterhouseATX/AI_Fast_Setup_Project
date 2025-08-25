# ================================
# app.py — Streamlit + Snowpark
# Dashboard + Custom Chart + AI
# ================================

# ----- Imports -----
import base64
import re
import pandas as pd
import plotly.express as px
import streamlit as st
from snowflake.snowpark import Session


# ================================
# Snowflake connection (Secrets-first; CSV fallback)
# ================================
def connect_snowflake_from_secrets() -> Session | None:
    cfg = st.secrets.get("snowflake")
    if not cfg:
        return None

    params = {
        "user":      cfg["user"],
        "role":      cfg.get("role"),
        "warehouse": cfg["warehouse"],
        "database":  cfg["database"],
        "schema":    cfg["schema"],
    }

    # Accept host or account; if host is given, also derive account
    host = cfg.get("host")
    acct = cfg.get("account")
    if host:
        params["host"] = host
        # Derive "bxb73724.us-west-2" from host "bxb73724.us-west-2.snowflakecomputing.com"
        if not acct and ".snowflakecomputing.com" in host:
            params["account"] = host.split(".snowflakecomputing.com", 1)[0]
    elif acct:
        params["account"] = acct
    else:
        st.error("Add either 'host' or 'account' to [snowflake] secrets.")
        st.stop()

    # Auth (password or key-pair)...
    # ... (keep the rest of your function the same)
    return Session.builder.configs(params).create()


# ----- Page Title -----
st.markdown(
    '<h1 style="text-align:center;color:#1f77b4;">📊 Sentiment Analysis + 🤖 AI Assistant</h1>',
    unsafe_allow_html=True
)

# ----- Sidebar: Data + Analysis -----
st.sidebar.header("Data Configuration")
# Defaults: from secrets if present, else sensible fallbacks for demo
default_db     = (st.secrets.get("snowflake", {}).get("database")  if not DEMO_MODE else "AVALANCHE_DB")
default_schema = (st.secrets.get("snowflake", {}).get("schema")    if not DEMO_MODE else "AVALANCHE_SCHEMA")

db     = st.sidebar.text_input("Database", value=default_db or "AVALANCHE_DB")
schema = st.sidebar.text_input("Schema",   value=default_schema or "AVALANCHE_SCHEMA")
table  = st.sidebar.text_input("Table",    value="REVIEW_SENTIMENT")

st.sidebar.subheader("Analysis")
load_all  = st.sidebar.checkbox("Load all rows", value=True, disabled=DEMO_MODE)
max_rows  = st.sidebar.slider("Max rows (ignored if Load all)", 100, 200000, 10000, step=500, disabled=DEMO_MODE)

# Core columns in your table / CSV
text_col  = st.sidebar.text_input("Text column", value="REVIEW_TEXT")
id_col    = st.sidebar.text_input("ID column (optional)", value="ORDER_ID")
score_col = st.sidebar.text_input("Score column (sentiment)", value="SENTIMENT_SCORE")
label_col = st.sidebar.text_input("Label column (optional)", value="SENTIMENT_LABEL")
pos_thr   = st.sidebar.number_input("Positive threshold", value=0.10, step=0.05, format="%.2f")
neg_thr   = st.sidebar.number_input("Negative threshold", value=-0.10, step=0.05, format="%.2f")

st.sidebar.markdown("---")
st.sidebar.header("AI Assistant")
ai_model  = st.sidebar.selectbox("Cortex model", ["llama3-8b", "mistral-7b", "snowflake-arctic"], index=0, disabled=DEMO_MODE)
ai_limit  = st.sidebar.number_input("Add LIMIT if missing", min_value=1, value=200, step=50, disabled=DEMO_MODE)

# Optional: Tabs UI (classic look) — tabs reset to first on rerun
use_tabs = st.sidebar.checkbox("Use tabs UI (may reset on rerun)", value=False)

# Context button only when connected to Snowflake
if not DEMO_MODE:
    if st.sidebar.button("Use context (DB, Schema)"):
        try:
            session.sql(f'USE DATABASE "{db}"').collect()
            session.sql(f'USE SCHEMA "{db}"."{schema}"').collect()
            st.sidebar.success(f"Context set to {db}.{schema}")
        except Exception as e:
            st.sidebar.error(f"Failed to set context: {e}")
else:
    st.sidebar.info("Running in demo (CSV) mode. Connect Snowflake via Secrets to enable SQL & AI.")

# Fully-qualified table (used when session is available)
fq = f'"{db}"."{schema}"."{table}"'

# Demo CSV controls (shown only in demo mode)
csv_file = None
if DEMO_MODE:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Demo data (CSV)")
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    default_csv_path = st.sidebar.text_input("Or path in repo", value="data/review_sentiment.csv")
    if uploaded:
        csv_file = uploaded  # file-like
    else:
        csv_file = default_csv_path  # string path

# ----- Data Loader -----
@st.cache_data(show_spinner=False)
def load_data_sql(session: Session, fq_table: str, text_c: str, score_c: str, label_c: str | None,
                  all_rows: bool, limit_n: int, pos_thr: float, neg_thr: float) -> pd.DataFrame:
    limit_clause = "" if all_rows else f"LIMIT {int(limit_n)}"
    sql = f'SELECT * FROM {fq_table} {limit_clause}'
    df = session.sql(sql).to_pandas()
    return _postprocess_df(df, text_c, score_c, label_c, pos_thr, neg_thr)

@st.cache_data(show_spinner=False)
def load_data_csv(src, text_c: str, score_c: str, label_c: str | None,
                  pos_thr: float, neg_thr: float) -> pd.DataFrame:
    df = pd.read_csv(src)
    return _postprocess_df(df, text_c, score_c, label_c, pos_thr, neg_thr)

def _postprocess_df(df: pd.DataFrame, text_c: str, score_c: str, label_c: str | None,
                    pos_thr: float, neg_thr: float) -> pd.DataFrame:
    # Validate required columns
    needed = [text_c, score_c]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}. Available: {list(df.columns)}")

    df = df.copy()
    df["polarity"] = df[score_c]
    if label_c and label_c in df.columns:
        df["sentiment"] = df[label_c].astype(str)
    else:
        def to_label(x: float) -> str:
            if x > pos_thr: return "Positive"
            if x < neg_thr: return "Negative"
            return "Neutral"
        df["sentiment"] = df["polarity"].apply(to_label)

    # Parse likely date columns if present
    for c in ["REVIEW_DATE", "SHIPPING_DATE"]:
        if c in df.columns:
            try:
                df[c] = pd.to_datetime(df[c], errors="coerce")
            except Exception:
                pass
    return df

# ----- Load Data -----
with st.spinner("Loading data..."):
    try:
        if DEMO_MODE:
            df = load_data_csv(csv_file, text_col, score_col, label_col if label_col else None, pos_thr, neg_thr)
        else:
            df = load_data_sql(session, fq, text_col, score_col, label_col if label_col else None,
                               load_all, max_rows, pos_thr, neg_thr)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

st.success(f"Loaded {len(df):,} rows from {'CSV demo' if DEMO_MODE else f'{db}.{schema}.{table}'}")


# ----- Top Navigation (sticky across reruns) -----
VIEWS = ["📈 Dashboard", "🧰 Custom Chart"] + ([] if DEMO_MODE else ["🤖 AI Assistant"])
if "active_view" not in st.session_state:
    st.session_state.active_view = VIEWS[0]

view = st.radio(
    "View", VIEWS,
    index=VIEWS.index(st.session_state.active_view),
    horizontal=True,
    key="active_view"
)


# ==========================
# Section: Dashboard
# ==========================
def render_dashboard():
    total = len(df)
    avg   = float(df["polarity"].mean())
    pos   = int((df["sentiment"] == "Positive").sum())
    neg   = int((df["sentiment"] == "Negative").sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total", f"{total:,}")
    c2.metric("Avg Polarity", f"{avg:.3f}")
    c3.metric("Positive", f"{pos:,}", f"{pos/total*100:.1f}%")
    c4.metric("Negative", f"{neg:,}", f"{neg/total*100:.1f}%")

    st.markdown("---")
    cc1, cc2 = st.columns(2)
    with cc1:
        counts = df["sentiment"].value_counts()
        fig = px.pie(values=counts.values, names=counts.index, title="Sentiment Distribution")
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)
    with cc2:
        fig = px.histogram(df, x="polarity", nbins=20, title="Polarity Score Distribution")
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Detailed Sentiment")
    chosen = st.multiselect(
        "Filter by sentiment",
        options=sorted(df["sentiment"].unique()),
        default=sorted(df["sentiment"].unique())
    )
    view_df = df[df["sentiment"].isin(chosen)]

    cols = [text_col, "sentiment", "polarity"]
    if id_col and id_col in df.columns:
        cols = [id_col] + cols
    if "PRODUCT" in df.columns:
        cols = ["PRODUCT"] + cols

    st.dataframe(
        view_df[cols].sort_values("polarity", ascending=False),
        use_container_width=True,
        height=420
    )

    st.markdown("---")
    st.subheader("Export")
    st.download_button(
        "Download CSV (current view)",
        data=view_df[cols].to_csv(index=False).encode(),
        file_name="sentiment_results.csv",
        mime="text/csv"
    )


# ==========================
# Section: Custom Chart
# ==========================
def render_custom_chart():
    st.subheader("Custom Chart")

    # Columns to hide from axis selectors
    exclude_from_dropdown = {"ORDER_ID", "TRACKING_NUMBER", "REVIEW_TEXT", "FILENAME"}

    # Work on a copy just for this chart
    work = df.copy()

    # Optional product filter (default = ALL)
    if "PRODUCT" in work.columns:
        all_prods = sorted(work["PRODUCT"].dropna().astype(str).unique().tolist())
        chart_products = st.multiselect(
            "Products to include (optional)",
            options=all_prods,
            default=[],   # EMPTY default => include ALL products
            help="Leave empty to include all products."
        )
        if chart_products:
            work = work[work["PRODUCT"].astype(str).isin(chart_products)]

    # Axis choices
    all_cols = [c for c in work.columns if c not in exclude_from_dropdown]
    numeric_cols = [c for c in work.select_dtypes(include="number").columns if c in all_cols]
    y_options = ["Count of rows"] + sorted(list(dict.fromkeys(numeric_cols + ["polarity"])))

    default_x = (
        "REVIEW_DATE" if "REVIEW_DATE" in all_cols else
        ("SHIPPING_DATE" if "SHIPPING_DATE" in all_cols else
         ("PRODUCT" if "PRODUCT" in all_cols else all_cols[0]))
    )
    x_col = st.selectbox("X axis", options=all_cols, index=all_cols.index(default_x) if default_x in all_cols else 0)
    y_sel = st.selectbox("Y axis", options=y_options, index=0)  # default "Count of rows"

    agg_map = {"count": "count", "mean": "mean", "sum": "sum", "median": "median"}
    agg_choice = st.selectbox("Aggregation", list(agg_map), index=(0 if y_sel=="Count of rows" else 1))
    chart_type = st.selectbox("Chart type", ["line", "bar", "scatter"], index=1 if y_sel=="Count of rows" else 0)

    # Clear grouping control
    group_by_product = st.checkbox(
        "Group / color by PRODUCT",
        value=False,  # DEFAULT OFF: all products aggregate together
        help="Turn on to compare products side-by-side. Leave off to combine all products into one series."
    )

    # Aggregate
    def aggregate(work_df, x, y_sel, agg, by_product=False):
        group_keys = [x]
        if by_product and "PRODUCT" in work_df.columns and "PRODUCT" not in group_keys:
            group_keys.append("PRODUCT")

        if y_sel == "Count of rows":
            tmp = work_df[group_keys].copy()
            tmp["value"] = 1
            out = tmp.groupby(group_keys, dropna=False)["value"].sum().reset_index()
        else:
            y_col = y_sel if y_sel in work_df.columns else "polarity"
            cols = list(dict.fromkeys(group_keys + [y_col]))
            tmp = work_df[cols].copy().rename(columns={y_col: "value"})
            out = tmp.groupby(group_keys, dropna=False)["value"].agg(agg_map[agg]).reset_index()

        # Nice sort for time/number X
        if pd.api.types.is_datetime64_any_dtype(out[x]) or pd.api.types.is_numeric_dtype(out[x]):
            out = out.sort_values(x)
        return out

    plot_df = aggregate(work, x_col, y_sel, agg_choice, by_product=group_by_product)

    # Draw
    title_txt = f"{y_sel} by {x_col}" + (" (by PRODUCT)" if group_by_product else "")
    color_arg = "PRODUCT" if (group_by_product and "PRODUCT" in plot_df.columns) else None

    if chart_type == "line":
        fig = px.line(plot_df, x=x_col, y="value", color=color_arg, title=title_txt)
    elif chart_type == "bar":
        fig = px.bar(plot_df, x=x_col, y="value", color=color_arg, barmode="group", title=title_txt)
    else:
        fig = px.scatter(plot_df, x=x_col, y="value", color=color_arg, title=title_txt)

    if color_arg:
        fig.update_layout(legend_title_text="PRODUCT")
    fig.update_layout(height=460)
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Tip: leave the product picker empty to include ALL products. Toggle “Group / color by PRODUCT” to split the series per product.")


# ==========================
# Section: AI Assistant
# ==========================
def render_ai_assistant():
    # Cortex via SQL (requires Snowflake session)
    def call_cortex_sql(model_name: str, prompt: str) -> str:
        prompt_sql = prompt.replace("'", "''")
        try:
            df_resp = session.sql(
                f"SELECT SNOWFLAKE.CORTEX.COMPLETE('{model_name}', '{prompt_sql}') AS RESPONSE"
            ).to_pandas()
            return str(df_resp.iloc[0, 0])
        except Exception as e:
            return f"(Cortex error: {e})"

    def extract_sql_block(text: str):
        m = re.search(r"```sql\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
        return m.group(1).strip() if m else None

    def maybe_add_limit(sql: str, n: int) -> str:
        return sql if " limit " in sql.lower() else f"{sql.rstrip(';')} LIMIT {int(n)}"

    def build_prompt(question: str) -> str:
        # Hint the model to output fenced SQL when appropriate
        return (
            "You are a helpful Snowflake data assistant.\n"
            f"Use the table {db}.{schema}.{table} (fully-qualified as {fq}).\n"
            "When useful, answer with a single SQL SELECT that reads from the fully-qualified table. "
            "If you output SQL, format it in a fenced code block like:\n"
            "[```sql]\nSELECT ... FROM {fq}\n[```]\n"
            "Keep answers short.\n\n"
            f"User question: {question}"
        )

    st.subheader("AI Assistant")
    if "chat_ai" not in st.session_state:
        st.session_state.chat_ai = []  # [{"role": "user"|"assistant", "content": str}]

    for m in st.session_state.chat_ai:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    user_q = st.chat_input(f"Ask about {db}.{schema}.{table}…")
    if user_q:
        st.session_state.chat_ai.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.write(user_q)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                prompt = build_prompt(user_q)
                raw = call_cortex_sql(ai_model, prompt)

                sql = extract_sql_block(raw)
                if sql:
                    st.markdown("**Proposed SQL**")
                    st.code(sql, language="sql")
                    try:
                        sql_to_run = maybe_add_limit(sql, ai_limit)
                        df_res = session.sql(sql_to_run).to_pandas()
                        if df_res.empty:
                            st.info("Query returned no rows.")
                        else:
                            st.dataframe(df_res, use_container_width=True, height=420)
                    except Exception as e:
                        st.error(f"SQL error: {e}")
                else:
                    st.write(raw)

        st.session_state.chat_ai.append({"role": "assistant", "content": raw})

    st.caption(f"AI over: {db}.{schema}.{table} • Model: {ai_model}")


# ----- Render chosen view -----
if use_tabs:
    # Classic tabs look (resets to first tab on rerun)
    t1, t2, t3 = st.tabs(VIEWS) if not DEMO_MODE else st.tabs(VIEWS)  # VIEWS already hides AI if demo
    with t1:
        render_dashboard()
    with t2:
        render_custom_chart()
    if not DEMO_MODE:
        with t3:
            render_ai_assistant()
else:
    # Sticky top radio (no jumping on rerun)
    if view == "📈 Dashboard":
        render_dashboard()
    elif view == "🧰 Custom Chart":
        render_custom_chart()
    else:
        # Only possible when not DEMO_MODE because VIEWS hides AI otherwise
        render_ai_assistant()
