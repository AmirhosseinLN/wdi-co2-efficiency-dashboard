import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="WDI CO2 Efficiency Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: "Inter", "Segoe UI", sans-serif;
}

.stApp {
    background-color: #0b1220;
    color: #e5e7eb;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

.main-header {
    background: linear-gradient(135deg, #111827 0%, #0f172a 100%);
    padding: 1.4rem 1.6rem;
    border-radius: 18px;
    border: 1px solid rgba(148, 163, 184, 0.16);
    box-shadow: 0 10px 30px rgba(0,0,0,0.22);
    margin-bottom: 1.25rem;
}

.main-title {
    font-size: 2rem;
    font-weight: 750;
    color: #f8fafc;
    margin-bottom: 0.35rem;
}

.sub-text {
    font-size: 0.98rem;
    color: #cbd5e1;
    line-height: 1.6;
}

.sub-text a {
    color: #93c5fd !important;
    text-decoration: none;
    font-weight: 600;
}

.sub-text a:hover {
    color: #bfdbfe !important;
    text-decoration: underline;
}

[data-testid="stMetric"] {
    background: linear-gradient(135deg, #111827 0%, #1e293b 100%);
    border: 1px solid rgba(148, 163, 184, 0.14);
    padding: 0.9rem;
    border-radius: 14px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.18);
}

[data-testid="stSidebar"] {
    background-color: #0f172a;
    border-right: 1px solid rgba(148, 163, 184, 0.12);
}

div[data-testid="stDataFrame"] {
    border: 1px solid rgba(148, 163, 184, 0.14);
    border-radius: 12px;
    overflow: hidden;
}

hr {
    border: none;
    border-top: 1px solid rgba(148, 163, 184, 0.18);
    margin: 1rem 0 1.2rem 0;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <div class="main-title">WDI CO2 Efficiency Dashboard</div>
    <div class="sub-text">
        This dashboard evaluates whether countries emit more or less CO2 than expected,
        given their GDP per capita, population, year, income group, and region.
    </div>
    <div class="sub-text" style="margin-top: 0.45rem;">
        Created by
        <a href="https://www.linkedin.com/in/amirhossein-latifinavid-5923272a7" target="_blank">
            Amirhossein Latifi Navid
        </a>
    </div>
</div>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df_model = pd.read_parquet("dashboard_data.parquet")

    # اگر فایل parquet با اسم ستون‌های خام ذخیره شده باشد، اینجا rename می‌کنیم
    rename_map = {
        "NY.GDP.PCAP.CD": "gdp_per_capita",
        "EN.ATM.CO2E.PC": "co2_per_capita",
        "SP.POP.TOTL": "population"
    }
    df_model = df_model.rename(columns=rename_map)

    # اگر efficiency_label داخل فایل نبود، بساز
    if "efficiency_label" not in df_model.columns and "residual" in df_model.columns:
        df_model["efficiency_label"] = df_model["residual"].apply(
            lambda x: "Better than expected" if x < 0 else "Worse than expected"
        )

    return df_model

try:
    df_model = load_data()
except Exception as e:
    st.error(f"Could not load dashboard_data.parquet. Error: {e}")
    st.stop()

with st.sidebar:
    st.header("Filters")
    years = sorted(df_model["Year"].unique())
    selected_year = st.selectbox("Year", years, index=len(years) - 1)

    all_regions = sorted(df_model["Region"].dropna().unique())
    selected_regions = st.multiselect("Region", all_regions, default=all_regions)

    all_income = sorted(df_model["Income Group"].dropna().unique())
    selected_income = st.multiselect("Income Group", all_income, default=all_income)

filtered = df_model[
    (df_model["Year"] == selected_year) &
    (df_model["Region"].isin(selected_regions)) &
    (df_model["Income Group"].isin(selected_income))
].copy()

if filtered.empty:
    st.warning("No rows match the current filters.")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Countries shown", len(filtered))
col2.metric("Average residual", f"{filtered['residual'].mean():.3f}")
col3.metric("Most efficient", filtered.loc[filtered['residual'].idxmin(), 'Country Name'])
col4.metric("Most inefficient", filtered.loc[filtered['residual'].idxmax(), 'Country Name'])

st.markdown("---")

st.subheader("CO2 Efficiency by Country")

map_df = filtered.dropna(subset=["Country Code"]).copy()

fig_map = px.choropleth(
    map_df,
    locations="Country Code",
    color="residual",
    hover_name="Country Name",
    hover_data={
        "Year": True,
        "Region": True,
        "Income Group": True,
        "co2_per_capita": ':.2f',
        "gdp_per_capita": ':.2f',
        "population": ':.0f',
        "residual": ':.3f'
    },
    color_continuous_scale="RdBu_r",
    range_color=(
        map_df["residual"].quantile(0.05),
        map_df["residual"].quantile(0.95)
    ),
    projection="natural earth"
)

fig_map.update_traces(
    marker_line_color="rgba(255,255,255,0.35)",
    marker_line_width=0.35
)

fig_map.update_layout(
    template="plotly_dark",
    margin=dict(l=0, r=0, t=10, b=0),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    geo=dict(
        showframe=False,
        showcoastlines=True,
        coastlinecolor="rgba(255,255,255,0.25)",
        bgcolor="rgba(0,0,0,0)",
        showland=True,
        landcolor="#111827",
        showocean=True,
        oceancolor="#0f172a",
        lakecolor="#0f172a"
    ),
    coloraxis_colorbar=dict(
        title=dict(text="Residual")
    ),
    transition_duration=450
)

st.plotly_chart(fig_map, use_container_width=True)

left, right = st.columns(2)

with left:
    st.subheader("GDP vs CO2 Scatter")

    fig_scatter = px.scatter(
        filtered,
        x="log_gdp",
        y="log_co2",
        color="residual",
        hover_name="Country Name",
        hover_data=["Region", "Income Group", "Year"],
        size="population",
        color_continuous_scale="RdBu_r",
        range_color=(
            filtered["residual"].quantile(0.05),
            filtered["residual"].quantile(0.95)
        )
    )

    fig_scatter.update_traces(
        marker=dict(
            line=dict(width=0.4, color="rgba(255,255,255,0.55)"),
            opacity=0.82
        )
    )

    fig_scatter.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Log GDP per capita",
        yaxis_title="Log CO2 per capita",
        coloraxis_colorbar=dict(title=dict(text="Residual")),
        transition_duration=450
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

with right:
    st.subheader("Top 10 Best and Worst Performers")

    best10 = filtered.nsmallest(10, "residual")[["Country Name", "residual", "Region", "Income Group"]].copy()
    worst10 = filtered.nlargest(10, "residual")[["Country Name", "residual", "Region", "Income Group"]].copy()

    best10["group"] = "Best"
    worst10["group"] = "Worst"
    rank_df = pd.concat([best10, worst10], ignore_index=True)

    fig_bar = px.bar(
        rank_df,
        x="residual",
        y="Country Name",
        color="group",
        orientation="h",
        hover_data=["Region", "Income Group"],
        color_discrete_map={
            "Best": "#2563eb",
            "Worst": "#dc2626"
        }
    )

    fig_bar.update_layout(
        yaxis={"categoryorder": "total ascending"},
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Residual",
        yaxis_title="Country",
        legend_title_text="Group",
        transition_duration=450
    )

    st.plotly_chart(fig_bar, use_container_width=True)

st.subheader("Detailed Country Table")

show_cols = [
    "Country Name", "Country Code", "Year", "Region", "Income Group",
    "gdp_per_capita", "co2_per_capita", "population",
    "predicted_log_co2", "log_co2", "residual", "efficiency_label"
]

st.dataframe(
    filtered[show_cols].sort_values("residual"),
    use_container_width=True
)

st.subheader("Compare One Country Over Time")

country_options = sorted(df_model["Country Name"].dropna().unique())
selected_country = st.selectbox(
    "Country",
    country_options,
    index=country_options.index("Malta") if "Malta" in country_options else 0
)

country_hist = df_model[df_model["Country Name"] == selected_country].sort_values("Year")

fig_country = go.Figure()

fig_country.add_trace(go.Scatter(
    x=country_hist["Year"],
    y=country_hist["log_co2"],
    mode="lines+markers",
    name="Actual log CO2",
    line=dict(width=3, color="#60a5fa"),
    marker=dict(size=6)
))

fig_country.add_trace(go.Scatter(
    x=country_hist["Year"],
    y=country_hist["predicted_log_co2"],
    mode="lines+markers",
    name="Predicted log CO2",
    line=dict(width=3, dash="dash", color="#f87171"),
    marker=dict(size=6)
))

fig_country.update_layout(
    title=f"{selected_country}: Actual vs Predicted Emissions Over Time",
    xaxis=dict(title="Year"),
    yaxis=dict(title="Log CO2"),
    legend=dict(orientation="h"),
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    transition_duration=450
)

st.plotly_chart(fig_country, use_container_width=True)

st.markdown("""
---
**Notes**

- Negative residual means a country emits less CO2 than expected.
- Positive residual means a country emits more CO2 than expected.
- This dashboard is based on World Development Indicators data and a Random Forest model.
""")